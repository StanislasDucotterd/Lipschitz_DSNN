import torch
from torch.nn import Linear
import torch.nn.functional as F

class LipschitzLinear(Linear):
    def __init__(self, lipschitz: float, projection, in_features: int, out_features: int, bias: bool = True):
        
        super().__init__(in_features, out_features, bias)
        
        self.lipschitz = lipschitz
        self.projection = projection
        
    def forward(self, x):
        #Some projection methods need to also update the maximum eigenvector
        lipschitz_weight = self.projection(self.weight, self.lipschitz)
        return F.linear(x, lipschitz_weight, self.bias)