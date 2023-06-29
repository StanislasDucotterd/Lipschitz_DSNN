import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class AbsoluteValue(torch.nn.Module):

    def __init__(self):
        super(AbsoluteValue, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.abs(input)

class LipschitzPReLU(torch.nn.Module):

    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        self.num_parameters = num_parameters
        super(LipschitzPReLU, self).__init__()
        if init == 'maxmin':
            prelu_weight = torch.empty(num_parameters)
            prelu_weight[::2] = 1
            prelu_weight[1::2] = -1
            self.prelu_weight = nn.Parameter(prelu_weight)
        else:
            self.prelu_weight = nn.Parameter(torch.empty(num_parameters).fill_(init))

    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input, torch.clip(self.prelu_weight, -1, 1))

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)