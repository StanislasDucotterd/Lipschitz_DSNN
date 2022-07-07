import torch
import torch.nn.functional as F

"""Some projection can be more efficient with knowledge of the previous
largets eigenvector and by being modified towards the end"""

def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-10)
    normalized_tensor = tensor / norm
    return normalized_tensor

def identity(weights, lipschitz_goal, additional_parameters=None):
    """no projection"""
    current_lipschitz = None
    new_weights = weights
    return weights, additional_parameters

def l1_normalization_conv(weights, lipschitz_goal, additional_parameters=None):
    """divides the conv layer by its L1 norm"""

    current_lipschitz = torch.max(torch.sum(torch.abs(weights), dim=(0, 2, 3)))
    new_weights = lipschitz_goal * weights / current_lipschitz
    return new_weights, additional_parameters

def l1_projection_conv(weights, lipschitz_goal, additional_parameters=None):
    """divides every column by its L1 norm"""
    current_lipschitzs = torch.sum(torch.abs(weights), dim=(0, 2, 3)).reshape(1, weights.shape[1], 1, 1)
    new_weights = lipschitz_goal * weights / current_lipschitzs
    return new_weights, additional_parameters

def linf_normalization_conv(weights, lipschitz_goal, additional_parameters=None):
    """divides the conv layer by its Linf norm"""

    current_lipschitz = torch.max(torch.sum(torch.abs(weights), dim=(1, 2, 3)))
    new_weights = lipschitz_goal * weights / current_lipschitz
    return new_weights, additional_parameters

def linf_projection_conv(weights, lipschitz_goal, additional_parameters=None):
    """divides every row by its L1 norm"""
    current_lipschitzs = torch.sum(torch.abs(weights), dim=(0, 2, 3)).reshape(weights.shape[0], 1, 1, 1)
    new_weights = lipschitz_goal * weights / current_lipschitzs
    return new_weights, additional_parameters

def spectral_norm_conv(weights, lipschitz_goal, additional_parameters):
    """divides the conv layer by its L2 norm"""
    kernel_size = weights.shape[2]
    padding = kernel_size //2

    u = additional_parameters['largest_eigenvector']
    if additional_parameters['end_of_epoch']: n_steps = 5
    else: n_steps = 1

    with torch.no_grad():
        for _ in range(n_steps):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            v = normalize(F.conv2d(u.flip(2,3), weights.permute(1, 0, 2, 3), padding=padding)).flip(2, 3)
            u = normalize(F.conv2d(v, weights, padding=padding))
            u = u.clone()
            v = v.clone()

    current_lipschitz = torch.sum(u * F.conv2d(v, weights, padding=padding))
    new_weights = lipschitz_goal * weights / current_lipschitz
    additional_parameters['largest_eigenvector'] = u
    return new_weights, additional_parameters