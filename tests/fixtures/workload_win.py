"""A fusion opportunity used only by the second performance workload."""

import mlx.core as mx
import mlx.nn as nn


class Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = mx.random.normal((1024,))
        self.b = mx.random.normal((1024,))

    def __call__(self, x):
        y = mx.maximum(x * 2.0 + self.a, 0.0)
        y = mx.minimum(y * x + self.b, 8.0)
        return (y - 1.0) * 0.5


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(21)
        self.chain = Chain()

    def __call__(self, x):
        return x + 1.0 if x.shape[0] == 1 else self.chain(x)


def build():
    return Model()
