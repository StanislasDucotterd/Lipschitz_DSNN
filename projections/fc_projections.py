import torch

def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-10)
    normalized_tensor = tensor / norm
    return normalized_tensor

def spectral_norm_fc(weights, lipschitz_goal, largest_eigenvector, end_of_training):

    current_lipschitz = torch.linalg.matrix_norm(weights, 2)
    new_weights = lipschitz_goal * weights / current_lipschitz

    return new_weights

def spectral_norm_fc_power_iter(weights, lipschitz_goal, largest_eigenvector, end_of_training):

    if end_of_training: n_iter = 5 
    else: n_iter = 1
    
    u = largest_eigenvector
    with torch.no_grad():
        for i in range(n_iter):
            v = normalize(weights.T @ u)
            u = normalize(weights @ v)
            u = u.clone()
            v = v.clone()
    
    current_lipschitz = u.T @ weights @ v
    new_weights = lipschitz_goal * weights / current_lipschitz

    return new_weights, u