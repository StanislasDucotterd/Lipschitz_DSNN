import torch
import torch.nn as nn
from torch.nn import Parameter

class ActNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(ActNorm, self).__init__()
        self.eps = eps
        self.dim = dim
        self._log_scale = Parameter(torch.Tensor(dim))
        self._shift = Parameter(torch.Tensor(dim))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :]

    def shift(self):
        return self._shift[None, :]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.dim == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.dim, -1).mean(dim=1)
                zero_mean = x - mean[None, :]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.dim, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() 
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())