import torch.nn as nn

class SimpleGenerator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=62, output_channel=1, output_size=28):
        super().__init__()
        self.input_dim = input_dim
        self.output_channel = output_channel
        self.output_size = output_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (output_size // 4) * (output_size // 4)),
            nn.BatchNorm1d(128 * (output_size // 4) * (output_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_channel, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, (self.output_size // 4), (self.output_size // 4))
        x = self.deconv(x)

        return x