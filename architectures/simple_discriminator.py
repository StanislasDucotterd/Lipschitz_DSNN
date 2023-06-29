"""
This GAN implementation is a replica of the the WGAN implementation that can be found in the same directory.
The only differences are that every layer in the discriminator is replaced with its strictly Lipschitz counterpart
(the original ones are commented out for readability) and weight clipping is removed, as the resulting model will be
strictly Lipschitz already.
"""


import torch
import numpy as np
import torch.nn as nn

from layers.BCOP.bcop import BCOP
from architectures.base_model import BaseModel
from layers.lipschitzlinear import LipschitzLinear
from layers.lipschitz_downsampling import LipschitzDownsample
from projections.fc_projections import bjorck_orthonormalize_fc

class SimpleDiscriminator(BaseModel):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, **params):
        super().__init__(**params)
        self.input_dim = 1
        self.output_dim = 1
        self.input_size = 28

        self.conv = nn.Sequential(
            LipschitzDownsample(self.input_dim, 64),
            self.init_activation(('conv', 64), bias=False),
            LipschitzDownsample(64, 128),
            self.init_activation(('conv', 128), bias=False),
        )

        self.fc = nn.Sequential(
            LipschitzLinear(1, bjorck_orthonormalize_fc, 128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            self.init_activation(('fc', 1024), bias=False),
            LipschitzLinear(1, bjorck_orthonormalize_fc, 1024, self.output_dim),
        )
        self.initialization('gan')

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x