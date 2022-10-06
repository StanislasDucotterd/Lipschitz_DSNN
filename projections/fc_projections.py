import torch
import numpy as np

def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-10)
    normalized_tensor = tensor / norm
    return normalized_tensor

def identity(weights, lipschitz_goal):
    """no projection"""
    return weights

def l2_normalization_fc(weights, lipschitz_goal):
    current_lipschitz = torch.linalg.matrix_norm(weights, 2)
    new_weights = lipschitz_goal * weights / current_lipschitz

    return new_weights

def bjorck_orthonormalize_fc(weights, lipschitz_goal, beta=0.5, iters=15):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    We only use the order 1.
    """
    w = weights / np.sqrt(weights.shape[0] * weights.shape[1])
    for _ in range(iters):
        w_t_w = w.t().mm(w)
        w = (1 + beta) * w - beta * w.mm(w_t_w)
    new_weights = lipschitz_goal * w

    return w