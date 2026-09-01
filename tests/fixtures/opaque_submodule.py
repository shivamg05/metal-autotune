"""Fixture: a compiled lambda held on the model. Nothing outside the model
can reach it by import path, so it records as an unnamed opaque call, no
region may include it, and no wrapper can replay a scope that contains it."""

import mlx.core as mx
import mlx.nn as nn


class PartOpaque(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(19)
        self.w = mx.random.normal((8, 8))
        self.fast = mx.compile(lambda a: mx.tanh(a) * 2.0)

    def __call__(self, x):
        y = x @ self.w
        z = self.fast(y)
        return z + 1.0


def build():
    return PartOpaque()
