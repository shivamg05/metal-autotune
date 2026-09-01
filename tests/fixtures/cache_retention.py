"""Fixture: retained AND consumed. The cache keeps a reference to k that no
wrapper can ever swap, while a later recorded op also consumes k. The freeze
must mark k python_retained; regions must end before k or leave it alone."""

import mlx.core as mx
import mlx.nn as nn


class Cached(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(15)
        self.wk = mx.random.normal((16, 16))
        self.wq = mx.random.normal((16, 16))
        self.cache = []

    def __call__(self, x):
        k = x @ self.wk
        self.cache.append(k)
        q = x @ self.wq
        return q @ k.T


def build():
    return Cached()
