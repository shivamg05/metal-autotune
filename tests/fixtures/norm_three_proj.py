"""Fixture: rms_norm feeding three projections that share its output. Chain
growth must merge neighbors on the shared input even with no data edge between
the projections."""

import mlx.core as mx
import mlx.nn as nn


class NormProj(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(13)
        self.g = mx.ones((32,))
        self.wq = mx.random.normal((32, 32))
        self.wk = mx.random.normal((32, 32))
        self.wv = mx.random.normal((32, 32))

    def __call__(self, x):
        h = mx.fast.rms_norm(x, self.g, eps=1e-5)
        return h @ self.wq, h @ self.wk, h @ self.wv


def build():
    return NormProj()
