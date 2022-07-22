import torch

def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-10)
    normalized_tensor = tensor / norm
    return normalized_tensor

def identity(weights, lipschitz_goal):
    """no projection"""
    return weights

def l1_normalization_fc(weights, lipschitz_goal):
    current_lipschitz = torch.linalg.matrix_norm(weights, 1)
    new_weights = lipschitz_goal * weights / current_lipschitz

    return new_weights

def l1_projection_fc(weights, lipschitz_goal):
    column_norms = torch.linalg.vector_norm(weights, 1, axis=0).unsqueeze(0)
    new_weights = weights / torch.clip(column_norms / lipschitz_goal, 1)

    return new_weights

def linf_normalization_fc(weights, lipschitz_goal):
    current_lipschitz = torch.linalg.matrix_norm(weights, float('inf'))
    new_weights = lipschitz_goal * weights / current_lipschitz

    return new_weights

def linf_projection_fc(weights, lipschitz_goal):
    row_norms = torch.linalg.vector_norm(weights, 1, axis=1).unsqueeze(1)
    new_weights = weights / torch.clip(row_norms / lipschitz_goal, 1)

    return new_weights

def l2_normalization_fc(weights, lipschitz_goal):
    current_lipschitz = torch.linalg.matrix_norm(weights, 2)
    new_weights = lipschitz_goal * weights / current_lipschitz

    return new_weights

def spectral_norm_fc(weights, lipschitz_goal, additional_parameters):

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