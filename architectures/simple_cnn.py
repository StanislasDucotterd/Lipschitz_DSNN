import torch
import torch.nn as nn
from architectures.base_model import BaseModel
from layers.lipschitzconv2d import LipschitzConv2d
from projections.conv_projections import (identity, l1_normalization_conv, l1_projection_conv, \
                    linf_normalization_conv, linf_projection_conv, spectral_norm_conv)


class SimpleCNN(BaseModel):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, **params):
        
        super().__init__(**params)

        modules = nn.ModuleList()

        if network_parameters['projection'] == 'no_projection':
            projection = identity
        elif network_parameters['projection'] == 'l1_norm':
            projection = l1_normalization_conv
        elif network_parameters['projection'] == 'l1_proj':
            projection = l1_projection_conv
        elif network_parameters['projection'] == 'linf_norm':
            projection = linf_normalization_conv
        elif network_parameters['projection'] == 'linf_proj':
            projection = linf_projection_conv
        elif network_parameters['projection'] == 'spectral_norm':
            projection = spectral_norm_conv
        else:
            raise ValueError('Projection type is not valid')

        num_channels = network_parameters['num_channels']
        lipschitz = network_parameters['lipschitz_constant']
        kernel_size = network_parameters['kernel_size']
        signal_size = network_parameters['signal_size']
        num_layers = network_parameters['num_layers']

        # First block
        modules.append(LipschitzConv2d(lipschitz, projection, 1, num_channels, kernel_size, signal_size))
        modules.append(self.init_activation(('conv', num_channels), bias=False))

        # Middle blocks
        for i in range(num_layers-2):
            modules.append(LipschitzConv2d(lipschitz, projection, num_channels, num_channels, kernel_size, signal_size))
            modules.append(self.init_activation(('conv', num_channels), bias=False))

        # Last block
        modules.append(LipschitzConv2d(lipschitz, projection, num_channels, 1, kernel_size, signal_size))

        self.initialization(init_type=network_parameters['weight_initialization'])
        self.num_params = self.get_num_params()

        self.layers = nn.Sequential(*modules)
        

    def forward(self, x):
        """ """
        return self.layers(x)
    
    
    def set_end_of_training(self):
        for i, module in enumerate(self.layers):
            if isinstance(module, LipschitzConv2d):
                self.layers[i].set_end_of_training()