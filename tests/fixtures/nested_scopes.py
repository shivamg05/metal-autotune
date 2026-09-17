"""Two weighted child modules separated by a parent-owned fusion target."""

import mlx.core as mx
import mlx.nn as nn


class Projection(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.weight = mx.full((16,), factor, dtype=mx.float32)

    def __call__(self, x):
        return x * self.weight


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = Projection(2.0)
        self.out = Projection(0.5)

    def __call__(self, x):
        x = self.proj(x)
        x = mx.maximum(x, 0.0)
        x = mx.minimum(x, 8.0)
        return self.out(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()

    def __call__(self, x):
        return self.block(x)


def build():
    return Model()
