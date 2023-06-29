import torch.nn as nn
from architectures.base_model import BaseModel

class SimpleGenerator(BaseModel):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, **params):
        super().__init__(**params)
        self.input_dim = 62
        self.output_channel = 1
        self.output_size = 28

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            self.init_activation(('fc', 1024), bias=False),
            nn.Linear(1024, 128 * (self.output_size // 4) * (self.output_size // 4)),
            nn.BatchNorm1d(128 * (self.output_size // 4) * (self.output_size // 4)),
            self.init_activation(('fc', 128 * (self.output_size // 4) * (self.output_size // 4)), bias=False),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            self.init_activation(('conv', 64), bias=False),
            nn.ConvTranspose2d(64, self.output_channel, 4, 2, 1),
            nn.Tanh(),
        )
        self.initialization('gan')

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, (self.output_size // 4), (self.output_size // 4))
        x = self.deconv(x)

        return x