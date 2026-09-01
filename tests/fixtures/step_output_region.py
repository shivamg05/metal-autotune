"""Fixture: a region whose output is also a step output. The wrapper must
return the kernel's value; e2e reads it directly."""

import mlx.core as mx
import mlx.nn as nn


class TwoOut(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(16)
        self.w = mx.random.normal((16, 16))

    def __call__(self, x):
        h = mx.maximum(x @ self.w, 0.0)
        z = h * 2.0
        return z, h


def build():
    return TwoOut()
