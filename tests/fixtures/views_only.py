"""Fixture: a stretch that is nothing but views (reshape, transpose, squeeze)
between two real ops. Views absorb into regions but a views-only stretch is not
a region; there is no work in it."""

import mlx.core as mx
import mlx.nn as nn


class Viewy(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(14)
        self.w = mx.random.normal((8, 8))

    def __call__(self, x):
        y = x @ self.w
        v = y.reshape(2, 2, 2, 4).transpose(0, 2, 1, 3).reshape(1, 4, 8).squeeze(0)
        return v + 1.0


def build():
    return Viewy()
