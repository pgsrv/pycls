#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""AnyNet models."""

import torch
import torch.nn as nn
from pycls.core.config import cfg
from torch.nn import Module


# ------------------------------------------------------------------------------------ #
# Shortcuts for common torch.nn layers
# ------------------------------------------------------------------------------------ #


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)


def norm2d(w_in):
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=w_in, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)


def pool2d(_w_in, k, *, stride=1):
    """Helper for building a pool2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    return nn.MaxPool2d(k, stride=stride, padding=(k - 1) // 2)


def gap2d(_w_in):
    """Helper for building a gap2d layer."""
    return nn.AdaptiveAvgPool2d((1, 1))


def linear(w_in, w_out, *, bias=False):
    """Helper for building a linear layer."""
    return nn.Linear(w_in, w_out, bias=bias)


def activation():
    """Helper for building an activation layer."""
    return nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)


# ------------------------------------------------------------------------------------ #
# Complexity (cx) calculations
# ------------------------------------------------------------------------------------ #


def conv2d_cx(cx, w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Accumulates complexity of conv2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    flops += k * k * w_in * w_out * h * w // groups + (w_out if bias else 0)
    params += k * k * w_in * w_out // groups + (w_out if bias else 0)
    acts += w_out * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def norm2d_cx(cx, w_in):
    """Accumulates complexity of norm2d into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    params += 2 * w_in
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def pool2d_cx(cx, w_in, k, *, stride=1):
    """Accumulates complexity of pool2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    acts += w_in * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def gap2d_cx(cx, _w_in):
    """Accumulates complexity of gap2d into cx = (h, w, flops, params, acts)."""
    flops, params, acts = cx["flops"], cx["params"], cx["acts"]
    return {"h": 1, "w": 1, "flops": flops, "params": params, "acts": acts}


def linear_cx(cx, w_in, w_out, *, bias=False):
    """Accumulates complexity of linear into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    flops += w_in * w_out + (w_out if bias else 0)
    params += w_in * w_out + (w_out if bias else 0)
    acts += w_out
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


# ------------------------------------------------------------------------------------ #
# Shared blocks
# ------------------------------------------------------------------------------------ #


class SiLU(Module):
    """SiLU activation function (also known as Swish): x * sigmoid(x)."""
    # Note: will be part of Pytorch 1.7, at which point can remove this.

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se, activation_fun=activation):
        super(SE, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.f_ex = nn.Sequential(
            conv2d(w_in, w_se, 1, bias=True),
            activation_fun(),
            conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = gap2d_cx(cx, w_in)
        cx = conv2d_cx(cx, w_in, w_se, 1, bias=True)
        cx = conv2d_cx(cx, w_se, w_in, 1, bias=True)
        cx["h"], cx["w"] = h, w
        return cx
