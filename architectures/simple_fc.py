import torch
import torch.nn as nn
from architectures.base_model import BaseModel
from layers.lipschitzlinear import LipschitzLinear
from projections.fc_projections import (identity, l1_normalization_fc, l1_projection_fc, \
                    linf_normalization_fc, linf_projection_fc, l2_normalization_fc)


class SimpleFC(BaseModel):
    """simple architecture for a denoiser"""
    def __init__(self, network_parameters, **params):
        
        super().__init__(**params)

        modules = nn.ModuleList()

        if network_parameters['projection'] == 'no_projection':
            projection = identity
        elif network_parameters['projection'] == 'l1_norm':
            projection = l1_normalization_fc
        elif network_parameters['projection'] == 'l1_proj':
            projection = l1_projection_fc
        elif network_parameters['projection'] == 'linf_norm':
            projection = linf_normalization_fc
        elif network_parameters['projection'] == 'linf_proj':
            projection = linf_projection_fc
        elif network_parameters['projection'] == 'l2_norm':
            projection = l2_normalization_fc
        else:
            raise ValueError('Projection type is not valid')

        layer_sizes = network_parameters['layer_sizes']
        lipschitz = network_parameters['lipschitz_constant']

        if len(layer_sizes) == 2:
            #Only spline activation functions without the weights
            modules.append(self.init_activation(('fc', 1)))
        else:
            #First blocks
            for i in range(len(layer_sizes)-2):
                modules.append(LipschitzLinear(lipschitz, projection, layer_sizes[i], layer_sizes[i+1]))
                modules.append(self.init_activation(('fc', layer_sizes[i+1])))

            # Last block
            modules.append(LipschitzLinear(lipschitz, projection, layer_sizes[-2], layer_sizes[-1]))

        self.initialization(init_type=network_parameters['weight_initialization'])
        self.num_params = self.get_num_params()

        self.layers = nn.Sequential(*modules)
        

    def forward(self, x):
        """ """
        return self.layers(x)