import torch
import torch.nn as nn
from layers.BCOP.bcop import BCOP
from architectures.base_model import BaseModel


class SimpleCNN(BaseModel):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, **params):
        
        super().__init__(**params)

        modules = nn.ModuleList()

        num_channels = network_parameters['num_channels']
        kernel_size = network_parameters['kernel_size']

        for i in range(len(num_channels)-2):
            modules.append(BCOP(num_channels[i], num_channels[i+1], kernel_size, bias=network_parameters['bias']))
            modules.append(self.init_activation(('conv', num_channels[i+1]), bias=False))

        # Last block
        modules.append(BCOP(num_channels[-2], num_channels[-1], kernel_size, bias=network_parameters['bias']))
        self.num_params = self.get_num_params()

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        """ """
        return self.layers(x)