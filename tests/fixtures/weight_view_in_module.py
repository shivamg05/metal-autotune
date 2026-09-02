"""Fixture: a projection through a transposed weight inside the same module
as the op before it. The transpose is not a module boundary, so only the
weight-only anchor rule makes the projection a candidate on its own."""

import mlx.core as mx
import mlx.nn as nn


class InPlace(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(17)
        self.w = mx.random.normal((48, 32))

    def __call__(self, x):
        h = mx.exp(x)
        return h @ self.w.T


def build():
    return InPlace()
