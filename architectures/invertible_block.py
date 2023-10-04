import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from architectures.base_model import BaseModel
from layers.lipschitzlinear import LipschitzLinear
from utils.actnorm import ActNorm

class InvertibleBlock(BaseModel):
    def __init__(self, lipschitz: float, width: int, dim: int, nb_layer: int, projection, bias: bool = True, **params):
        
        super().__init__(**params)
        
        self.lipschitz = lipschitz
        self.actnorm = ActNorm(dim)
        modules = nn.ModuleList()

        modules.append(LipschitzLinear(1.0, projection, dim, width, bias))
        modules.append(self.init_activation(('fc', width)))

        for i in range(nb_layer-2):
            modules.append(LipschitzLinear(1.0, projection, width, width, bias))
            modules.append(self.init_activation(('fc', width)))

        modules.append(LipschitzLinear(1.0, projection, width, dim, bias))
        self.layers = nn.Sequential(*modules)
        
    def forward(self, x):
        #x, _ = self.actnorm(x)
        return x + self.lipschitz * self.layers(x)
    
    def inverse(self, y):
        x = y.clone()
        for _ in range(10):
            x = y - self.lipschitz * self.layers(x)
        #x = self.actnorm.inverse(x)
        return x