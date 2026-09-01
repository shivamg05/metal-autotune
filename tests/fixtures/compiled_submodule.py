"""Fixture: part of the forward runs under the model's own mx.compile. The
compiled function lives at module level, so the tracer can name it by import
path: it records as one opaque call, no region may include it, and a wrapper
on the scope around it can still replay the scope by calling that path."""

import mlx.core as mx
import mlx.nn as nn


@mx.compile
def fast_tanh(a):
    return mx.tanh(a) * 2.0


class PartCompiled(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(19)
        self.w = mx.random.normal((8, 8))

    def __call__(self, x):
        y = x @ self.w
        z = fast_tanh(y)
        return z + 1.0


class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.part = PartCompiled()

    def __call__(self, x):
        return self.part(x)


def build():
    return Outer()
