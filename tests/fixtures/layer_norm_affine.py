"""Fixture: layer_norm over the last axis with weight and bias present. The
kernel must center, normalize, then scale by weight and shift by bias."""

import mlx.core as mx
import mlx.nn as nn


class LayerNormAffine(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(17)
        self.w = mx.random.normal((32,))
        self.b = mx.random.normal((32,))

    def __call__(self, x):
        return mx.fast.layer_norm(x, self.w, self.b, 1e-5)


def build():
    return LayerNormAffine()
