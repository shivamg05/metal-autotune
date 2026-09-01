"""Fixture: calls .item() mid-forward, forcing evaluation during the recording
pass. Must record completely and trigger the in-pass-evaluation memory warning."""

import mlx.core as mx
import mlx.nn as nn


class Eager(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(17)
        self.w = mx.random.normal((8, 8))

    def __call__(self, x):
        y = x @ self.w
        scale = mx.abs(y).max().item()
        return y / (scale + 1.0)


def build():
    return Eager()
