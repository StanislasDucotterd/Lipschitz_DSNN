import torch

class Identity(Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input

class AbsoluteValue(Module):

    def __init__(self):
        super(AbsoluteValue, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.abs(input)