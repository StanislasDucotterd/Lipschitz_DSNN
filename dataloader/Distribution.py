import math
import torch
from torch.utils.data import Dataset
from architectures.simple_generator import SimpleGenerator

class BaseDistribution(torch.nn.Module):
    def __init__(self, dim, device):
        self.dim = dim
        self.device = device

class Dirac(BaseDistribution):
    def __init__(self, dim, center, device):
        super().__init__(dim, device)
        self.center = center

    def __call__(self, batch_size):
        X = torch.zeros(batch_size, self.dim, device=self.device)
        X = X + self.center

        return X

class L2Ball(BaseDistribution):
    """Sample uniformly on the L2 ball"""
    def __init__(self, dim, center, device):
        super().__init__(dim, device)
        self.center = center

    def __call__(self, batch_size):
        X = torch.randn(batch_size, self.dim, device=self.device)
        X = X / torch.linalg.norm(X, ord=2, dim=1).unsqueeze(1)
        X = X + self.center

        return X

class L1Ball(BaseDistribution):
    """Sample uniformly on the L1 ball"""
    def __init__(self, dim, center, device):
        super().__init__(dim, device)
        self.center = center
        self.distribution = torch.distributions.laplace.Laplace(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))

    def __call__(self, batch_size):
        X = self.distribution.sample((batch_size, self.dim)).squeeze()
        X = X / torch.linalg.norm(X, ord=1, dim=1).unsqueeze(1)
        X = X + self.center

        return X

class MnistGenerator(torch.nn.Module):
    """GAN Genrator of MNIST Images"""
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.input_dim = 62
        self.generator = SimpleGenerator(input_dim=self.input_dim, output_channel=1, output_size=28)
        self.generator.load_state_dict(torch.load('data/generator_weights.pth'))
        self.generator = self.generator.to(self.device)
        self.generator.eval()

    def __call__(self, batch_size):
        X = torch.rand(batch_size, self.input_dim, device=self.device)
        X = self.generator(X)
        X = (X + 1) / 2

        return X

class MNIST(Dataset):
    "Real MNIST Images"
    def __init__(self, data_file):
        super(Dataset, self).__init__()
        self.input, self.target = torch.load(data_file)
        self.input = self.input.unsqueeze(1)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx,:,:,:] / 255, self.target[idx]