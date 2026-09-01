"""Fixture: a deliberately unfused elementwise chain. Every op is a full
memory round trip in the library; one fused kernel reads x once and writes
once, so a correct fusion must beat the library by a wide margin. The chain
lives in a child module because that is the delivery scope a wrapper swaps."""

import mlx.core as mx
import mlx.nn as nn


class Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = mx.random.normal((1024,))
        self.b = mx.random.normal((1024,))

    def __call__(self, x):
        y = x * 2.0
        y = y + self.a
        y = mx.maximum(y, 0.0)
        y = y * x
        y = y + self.b
        y = mx.minimum(y, 8.0)
        y = y - 1.0
        return y * 0.5


class Planted(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(21)
        self.chain = Chain()

    def __call__(self, x):
        return self.chain(x)


def build():
    return Planted()
