import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractproperty, abstractmethod


def slope_clipping(cs, T):
    device = cs.device
    n = cs.shape[1]
    new_slopes = torch.clamp(cs[:,1:] - cs[:,:-1], -T, T)
    new_cs = torch.zeros(cs.shape, device=device)
    new_cs[:,1:] = torch.cumsum(new_slopes, dim=1)
    new_cs = new_cs + torch.mean(cs - new_cs, dim=1).unsqueeze(1)

    return new_cs

def initialize_coeffs(init, grid_tensor):
        """The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators)."""
        
        if init == 'identity':
            coefficients = grid_tensor
        elif init == 'relu':
            coefficients = F.relu(grid_tensor)
        elif init == 'absolute_value':
            coefficients = torch.abs(grid_tensor)
            
        elif init == 'maxmin':
            # initalize half of the activations with the absolute and the other half with the 
            # identity. This is similar to maxmin because max(x1, x2) = (x1 + x2)/2 + |x1 - x2|/2 
            # and min(x1, x2) = (x1 + x2)/2 - |x1 - x2|/2
            coefficients[::2, :] = (grid_tensor[::2, :]).abs()
            coefficients[1::2, :] = grid_tensor[1::2, :]

        elif init == 'max_tv':
            # initialize the spline such that its tv is maximized
            # while being 1-Lipschitz
            grid = grid_tensor[1] - grid_tensor[0]
            coefficients = torch.zeros(grid_tensor.shape)
            coefficients[::2,::2] = - grid / 2
            coefficients[::2,1::2] = grid / 2
            coefficients[1::2,::2] = grid / 2
            coefficients[1::2,1::2] = - grid / 2
        
        else:
            raise ValueError('init should be in [identity, relu, absolute_value, maxmin, sawtooth, max_tv].')
        
        return coefficients


class LinearSpline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients_vect, grid, range_, zero_knot_indexes, even):

        # The value of the spline at any x is a combination 
        # of at most two coefficients
        if even:
            x = x + grid / 2
        x_clamped = x.clamp(min=-range_, max=(range_ - grid.item()))

        floored_x = torch.floor(x_clamped / grid)  #left coefficient
        #fracs = x_clamped / grid - floored_x
        fracs = x / grid - floored_x  # distance to left coefficient

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1) + floored_x).long()

        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = coefficients_vect[indexes + 1] * fracs + \
            coefficients_vect[indexes] * (1 - fracs)

        ctx.save_for_backward(fracs, coefficients_vect, indexes, grid)
        return activation_output - grid / 2

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients_vect, indexes, grid = ctx.saved_tensors
        grad_x = (coefficients_vect[indexes + 1] -
                  coefficients_vect[indexes]) / grid * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (fracs * grad_out).view(-1))
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                            ((1 - fracs) * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None

class LinearSpline(ABC, nn.Module):
    """
    Class for LinearSpline activation functions

    Args:
        mode (str): 'conv' (convolutional) or 'fc' (fully-connected).
        num_activations (int) : number of activation functions
        size (int): number of coefficients of spline grid; the number of knots K = size - 2.
        range_ (float) : positive range of the B-spline expansion. B-splines range = [-range_, range_].
        init (str): Function to initialize activations as (e.g. 'relu', 'identity', 'absolute_value').
        lipschitz_constraint (bool): Constrain the activation to be 1-Lipschitz
    """

    def __init__(self, mode, num_activations, size, range_, init,
                 lipschitz_constraint, **kwargs):

        if mode not in ['conv', 'fc']:
            raise ValueError('Mode should be either "conv" or "fc".')
        if int(num_activations) < 1:
            raise TypeError('num_activations needs to be a '
                            'positive integer...')

        super().__init__()

        self.mode = mode
        self.size = int(size)
        self.even = self.size % 2 == 0
        self.num_activations = int(num_activations)
        self.init = init
        self.range_ = float(range_)
        grid = 2 * self.range_ / (self.size-1)
        self.grid = torch.Tensor([grid])

        self.init_zero_knot_indexes()
        self.D2_filter = Tensor([1, -2, 1]).view(1, 1, 3).div(self.grid)
        self.lipschitz_constraint = lipschitz_constraint

        # tensor with locations of spline coefficients
        self.grid_tensor = torch.linspace(-self.range_, self.range_, self.size).expand((self.num_activations, self.size))
        coefficients = initialize_coeffs(init, self.grid_tensor)  # spline coefficients
        # Need to vectorize coefficients to perform specific operations
        # size: (num_activations*size)
        self.coefficients_vect = nn.Parameter(coefficients.contiguous().view(-1))

        if (self.mode == 'fc'):
            self.scaling_coeffs_vect = nn.Parameter(torch.ones((1, self.num_activations)))
        else:
            self.scaling_coeffs_vect = nn.Parameter(torch.ones((1, self.num_activations, 1, 1)))

    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations)
        self.zero_knot_indexes = (activation_arange * self.size +
                                  (self.size // 2))

    @property
    def coefficients(self):
        """ B-spline coefficients. """
        return self.coefficients_vect.view(self.num_activations, self.size)
    
    @property
    def lipschitz_coefficients(self):
        """Projection of B-spline coefficients such that they are 1-Lipschitz"""
        return slope_clipping(self.coefficients, self.grid.item())
    
    @property
    def lipschitz_coefficients_vect(self):
        """Projection of B-spline coefficients such that they are 1-Lipschitz"""
        return self.lipschitz_coefficients.contiguous().view(-1)

    @property
    def relu_slopes(self):
        """ Get the activation relu slopes {a_k},
        by doing a valid convolution of the coefficients {c_k}
        with the second-order finite-difference filter [1,-2,1].
        """
        D2_filter = self.D2_filter.to(device=self.coefficients.device)

        if self.lipschitz_constraint:
            slopes = F.conv1d(self.lipschitz_coefficients.unsqueeze(1), D2_filter).squeeze(1)
        else:
            slopes = F.conv1d(self.coefficients.unsqueeze(1), D2_filter).squeeze(1)
        return slopes


    def forward(self, x):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        assert x.size(1) == self.num_activations, \
            'Wrong shape of input: {} != {}.'.format(input.size(1), self.num_activations)

        grid = self.grid.to(self.coefficients_vect.device)
        zero_knot_indexes = self.zero_knot_indexes.to(grid.device)

        x = x.mul(self.scaling_coeffs_vect)

        if self.lipschitz_constraint:
            x = LinearSpline_Func.apply(x, self.lipschitz_coefficients_vect, grid, 
                                            self.range_, zero_knot_indexes, self.even)

        else:
            x = LinearSpline_Func.apply(x, self.coefficients_vect, grid, 
                                             self.range_, zero_knot_indexes, self.even)

        x = x.div(self.scaling_coeffs_vect)
                                        
        return x


    def extra_repr(self):
        """ repr for print(model) """

        s = ('mode={mode}, num_activations={num_activations}, '
             'init={init}, size={size}, grid={grid[0]:.3f}, '
             'lipschitz_constraint={lipschitz_constraint}.')

        return s.format(**self.__dict__)

    def totalVariation(self, **kwargs):
        """
        Computes the second-order total-variation regularization.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        The regularization term applied to this function is:
        TV(2)(deepsline) = ||a||_1.
        """
        return self.relu_slopes.norm(1, dim=1)