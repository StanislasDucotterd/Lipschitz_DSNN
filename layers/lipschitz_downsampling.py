import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
from projections.fc_projections import bjorck_orthonormalize_fc

class LipschitzDownsample(_ConvNd):
    def __init__(self, in_channels, out_channels):
        """Module for Lipschitz constrained convolution layer"""
        
        padding = _pair(1)
        kernel_size = _pair(4)
        stride = _pair(2)
        dilation = _pair(1)
        transposed = False
        output_padding = _pair(0)
        groups = 1
        bias = True
        #we dont do padding but still need to specify it for _ConvNd
        padding_mode = 'circular' 
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, \
                         output_padding, groups, bias, padding_mode)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
            
        ortho_weights = bjorck_orthonormalize_fc(self.weight.reshape(self.out_channels, -1), 1, iters=12)
        ortho_weights = ortho_weights.reshape(self.out_channels, self.in_channels, 4, 4)
        return F.conv2d(x, ortho_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LipschitzDownsample2(_ConvNd):
    def __init__(self, in_channels, out_channels):
        """Module for Lipschitz constrained convolution layer"""
        
        padding = _pair(0)
        kernel_size = _pair(2)
        stride = _pair(2)
        dilation = _pair(1)
        transposed = False
        output_padding = _pair(0)
        groups = 1
        bias = True
        #we dont do padding but still need to specify it for _ConvNd
        padding_mode = 'zeros' 
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, \
                         output_padding, groups, bias, padding_mode)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
            
        ortho_weights = bjorck_orthonormalize_fc(self.weight.reshape(self.out_channels, self.in_channels*4), 1, iters=12)
        ortho_weights = ortho_weights.reshape(self.out_channels, self.in_channels, 2, 2)
        return F.conv2d(x, ortho_weights, self.bias, self.stride, _pair(0), self.dilation, self.groups)