import torch
import torch.nn as nn
from architectures.base_model import BaseModel
from architectures.invertible_block import InvertibleBlock
from projections.fc_projections import identity, bjorck_orthonormalize_fc


class InvResNet(BaseModel):
    """simple architecture for a fully-connected network"""
    def __init__(self, network_parameters, **params):
        
        super().__init__(**params)

        modules = nn.ModuleList()

        if network_parameters['projection'] == 'no_projection':
            projection = identity
        elif network_parameters['projection'] == 'orthonormalize':
            if 'bjorck_iter' in network_parameters:
                def proj(weights, lipschitz_goal):
                    return bjorck_orthonormalize_fc(weights, lipschitz_goal, beta=0.5, iters=network_parameters['bjorck_iter'])
                projection = proj
            else:
                projection = bjorck_orthonormalize_fc
        else:
            raise ValueError('Projection type is not valid')

        nb_block = network_parameters['nb_block']
        lipschitz = network_parameters['lipschitz']
        width = network_parameters['width']
        dim = network_parameters['dim']
        nb_layer = network_parameters['nb_layer']
        bias = network_parameters['bias']

        for i in range(nb_block):
            modules.append(InvertibleBlock(lipschitz, width, dim, nb_layer, projection, bias, **params))

        self.initialization(init_type=network_parameters['weight_initialization'])
        self.num_params = self.get_num_params()

        self.layers = nn.Sequential(*modules)
    

    def forward(self, x):
        """ """
        return self.layers(x)
    
    def inverse(self, x):
        """ """
        for i in range(len(self.layers), 0, -1):
            x = self.layers[i-1].inverse(x)
        return x