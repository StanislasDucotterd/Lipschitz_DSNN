import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import relu


class HouseHolder(nn.Module):
    def __init__(self, mode, channels):
        super(HouseHolder, self).__init__()
        assert (channels % 2) == 0
        eff_channels = channels // 2
        
        if mode == 'conv':
            self.theta = nn.Parameter(
                    0.5 * np.pi * torch.ones(1, eff_channels, 1, 1), requires_grad=True)
        else:
            self.theta = nn.Parameter(
                    0.5 * np.pi * torch.ones(1, eff_channels), requires_grad=True)


    def forward(self, z, axis=1):
        theta = self.theta
        x, y = z.split(z.shape[axis] // 2, axis)
                    
        selector = (x * torch.sin(0.5 * theta)) - (y * torch.cos(0.5 * theta))
        
        a_2 = x*torch.cos(theta) + y*torch.sin(theta)
        b_2 = x*torch.sin(theta) - y*torch.cos(theta)
        
        a = (x * (selector <= 0) + a_2 * (selector > 0))
        b = (y * (selector <= 0) + b_2 * (selector > 0))
        
        return torch.cat([a, b], dim=axis)