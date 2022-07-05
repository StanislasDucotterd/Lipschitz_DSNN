import torch
from torch import Tensor

class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input

class AbsoluteValue(torch.nn.Module):

    def __init__(self):
        super(AbsoluteValue, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.abs(input)