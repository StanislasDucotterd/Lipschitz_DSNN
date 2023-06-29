"""
Utility functions
"""

from einops import rearrange
import torch.nn.functional as F
from torch.nn.functional import normalize, conv_transpose2d, conv2d

import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict
from functools import reduce
from copy import deepcopy
import sys
import math

def power_iteration(A, init_u=None, n_iters=10, return_uv=True):
    """
    Power iteration for matrix
    """
    shape = list(A.shape)
    # shape[-2] = shape[-1]
    shape[-1] = 1
    shape = tuple(shape)
    if init_u is None:
        u = torch.randn(*shape, dtype=A.dtype, device=A.device)
    else:
        assert tuple(init_u.shape) == shape, (init_u.shape, shape)
        u = init_u
    for _ in range(n_iters):
        v = A.transpose(-1, -2) @ u
        v /= v.norm(dim=-2, keepdim=True)
        u = A @ v
        u /= u.norm(dim=-2, keepdim=True)
    s = (u.transpose(-1, -2) @ A @ v).squeeze(-1).squeeze(-1)
    if return_uv:
        return u, s, v
    return s

def cyclic_pad_2d(x, pads, even_h=False, even_w=False):
    """
    Implemenation of cyclic padding for 2-D image input
    """
    pad_change_h = -1 if even_h else 0
    pad_change_w = -1 if even_w else 0
    pad_h, pad_w = pads
    if pad_h != 0:
        v_pad = torch.cat((x[..., :, -pad_h:, :], x,
                           x[..., :, :pad_h+pad_change_h, :]), dim=-2)
    elif pad_change_h != 0:
        v_pad = torch.cat((x, x[..., :, :pad_change_h, :]), dim=-2)
    else:
        v_pad = x
    if pad_w != 0:
        h_pad = torch.cat((v_pad[..., :, :, -pad_w:],
                           v_pad, v_pad[..., :, :, :pad_w+pad_change_w]), dim=-1)
    elif pad_change_w != 0:
        h_pad = torch.cat((v_pad, v_pad[..., :, :, :+pad_change_w]), dim=-1)
    else:
        h_pad = v_pad
    return h_pad


def conv2d_cyclic_pad(
        x, weight, bias=None):
    """
    Implemenation of cyclic padding followed by a normal convolution
    """
    kh, kw = weight.size(-2), weight.size(-1)
    x = cyclic_pad_2d(x, [kh // 2, kw // 2], (kh % 2 == 0), (kw % 2 == 0))
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return F.conv2d(x, weight, bias)

def bjorck_orthonormalize(
        w, beta=0.5, iters=20, order=1, power_iteration_scaling=False,
        default_scaling=False):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """

    if w.shape[-2] < w.shape[-1]:
        return bjorck_orthonormalize(
            w.transpose(-1, -2),
            beta=beta, iters=iters, order=order,
            power_iteration_scaling=power_iteration_scaling,
            default_scaling=default_scaling).transpose(
            -1, -2)

    if power_iteration_scaling:
        with torch.no_grad():
            s = power_iteration(w, return_uv=False)
        w = w / s.unsqueeze(-1).unsqueeze(-1)
    elif default_scaling:
        w = w / ((w.shape[0] * w.shape[1]) ** 0.5)
    assert order == 1, "only first order Bjorck is supported"
    for _ in range(iters):
        w_t_w = w.transpose(-1, -2) @ w
        w = (1 + beta) * w - beta * w @ w_t_w
    return w

# The following two functions are directly taken from https://arxiv.org/pdf/1805.10408.pdf
def conv_singular_values_numpy(kernel, input_shape):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)

def conv_clip_2_norm_numpy(
        kernel, input_shape, clip_to, force_same=False, complex_conv=False,
        returns_full_conv=False):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    transform_coefficients = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    U, D, V = np.linalg.svd(transform_coefficients,
                            compute_uv=True, full_matrices=False)
    if force_same:
        D_clipped = np.ones_like(D) * clip_to
    else:
        D_clipped = np.minimum(D, clip_to)
    if kernel.shape[2] > kernel.shape[3]:
        clipped_transform_coefficients = np.matmul(U, D_clipped[..., None] * V)
    else:
        clipped_transform_coefficients = np.matmul(U * D_clipped[..., None, :], V)
    clipped_kernel = np.fft.ifft2(clipped_transform_coefficients, axes=[0, 1])
    if not complex_conv:
        clipped_kernel = clipped_kernel.real
    if not returns_full_conv:
        clipped_kernel = clipped_kernel[np.ix_(*[range(d) for d in kernel.shape])]
    clipped_kernel = np.transpose(clipped_kernel, [2, 3, 0, 1])
    return clipped_kernel

def default_collate_op(x, y):
    if x is None:
        return [y]
    if y is None:  # avoid appending None and nan
        return x
    if type(y) == list:
        x.extend(y)
    else:
        x.append(y)
    return x


def default_summarize_op(x, dtype):
    if dtype == "scalar":
        if len(x) == 0:
            return 0
        return sum(x) / len(x)
    if dtype == "histogram":
        return torch.tensor(x)
    return x


def default_display_op(x, dtype):
    if dtype == "scalar":
        return "{:.4f}".format(x)
    if dtype == "histogram":
        return "histogram[n={}]".format(len(x))
    return x


def prod(x):
    return reduce(lambda a, b: a * b, x)


class StreamlinedModule(nn.Module):
    def __init__(self):
        self.streamline = False
        super(StreamlinedModule, self).__init__()

    def set_streamline(self, streamline=False):
        self.streamline = streamline
        return streamline


def streamline_model(model, streamline=False):
    for m in model.modules():
        if isinstance(m, StreamlinedModule):
            m.set_streamline(streamline)


# Context manager that streamlines the module of interest in the context only
class Streamline:
    def __init__(self, module, new_flag=True, old_flag=False):
        self.module = module
        self.new_flag = new_flag
        self.old_flag = old_flag

    def __enter__(self):
        streamline_model(self.module, self.new_flag)

    def __exit__(self, *args, **kwargs):
        streamline_model(self.module, self.old_flag)


# A helper object for logging all the data
class Accumulator:
    def __init__(self):
        self.data = defaultdict(list)
        self.data_dtype = defaultdict(None)

    def __call__(
        self,
        name,
        value=None,
        dtype=None,
        collate_op=default_collate_op,
        summarize_op=None,
    ):
        if value is None:
            if summarize_op is not None:
                return summarize_op(self.data[name])
            return self.data[name]
        self.data[name] = default_collate_op(self.data[name], value)
        if dtype is not None:
            self.data_dtype[name] = dtype
        assert dtype == self.data_dtype[name]

    def summarize(self, summarize_op=default_summarize_op):
        for key in self.data:
            self.data[key] = summarize_op(self.data[key], self.data_dtype[key])

    def collect(self):
        return {key: self.__call__(key) for key in self.data}

    def filter(self, dtype=None, level=None, op=None):
        if op is None:
            op = lambda x: x
        if dtype is None:
            return self.collect()
        return {
            key: op(self.__call__(key))
            for key in filter(
                lambda x: self.data_dtype[x] == dtype
                and (x.count("/") <= level if (level is not None) else True),
                self.data,
            )
        }

    def latest_str(self):
        return ", ".join(
            "{}={:.4f}".format(key, value[-1] if len(value) > 0 else math.nan)
            for key, value in self.collect().items()
        )

    def summary_str(self, dtype=None, level=None):
        return ", ".join(
            "{}={}".format(
                key, default_display_op(self.__call__(key), self.data_dtype[key])
            )
            for key in self.filter(dtype=dtype, level=level)
        )

    def __str__(self):
        return self.summary_str()

# A logger that sync terminal output to a logger file
class Logger(object):
    def __init__(self, logdir):
        self.terminal = sys.stdout
        self.log = open(logdir, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s