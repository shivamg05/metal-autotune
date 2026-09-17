"""A parent whose fusion target starts inside one child and ends in the next,
with a third child in front that gets its own kernel first: the shape of a
FLUX feed-forward block (linear_in, SwiGLU, linear_out)."""

import mlx.core as mx
import mlx.nn as nn


class Projection(nn.Module):
    def __init__(self, width, factor):
        super().__init__()
        self.weight = mx.full((width,), factor, dtype=mx.float32)

    def __call__(self, x):
        return x * self.weight


class Gate(nn.Module):
    """silu(a) * b, as SwiGLU does. nn.silu ships compiled, so it records as
    one opaque call, and the multiply after it starts a fresh chain."""

    def __call__(self, x):
        a, b = mx.split(x, 2, axis=-1)
        return nn.silu(a) * b


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_in = Projection(16, 2.0)
        self.act = Gate()
        self.lin_out = Projection(8, 0.5)

    def __call__(self, x):
        return self.lin_out(self.act(self.lin_in(x)))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()

    def __call__(self, x):
        return self.block(x)


def build():
    return Model()
