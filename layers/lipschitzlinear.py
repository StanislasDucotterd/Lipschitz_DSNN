import torch
from torch.nn import Linear
import torch.nn.functional as F

def normalize(tensor):
    norm = float(torch.sqrt(torch.sum(tensor * tensor)))
    norm = max(norm, 1e-10)
    normalized_tensor = tensor / norm
    return normalized_tensor

class LipschitzLinear(Linear):
    def __init__(self, lipschitz: float, projection, in_features: int, out_features: int, bias: bool = True):
        
        super().__init__(in_features, out_features, bias)
        
        self.lipschitz = lipschitz
        self.projection = projection
        self.max_eigenvector = normalize(torch.randn(1, out_features))
        
    def forward(self, x):
        #Some projection methods need to also update the maximum eigenvector
        proj_weight = self.projection(self.weight, self.lipschitz, self.max_eigenvector)
        if type(proj_weight) is tuple:
            self.max_eigenvector = proj_weight[1]
            proj_weight = proj_weight[0]

        return F.linear(x, proj_weight, self.bias)