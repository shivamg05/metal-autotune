"""Fixture: a norm written out by hand (mean, rsqrt, scale), so its region
holds a reduction the scaffold must lower itself. The lowering's sum tree
adds in its own order, so the harness checks it within tolerance, never
bitwise; held to bitwise it fails the smoke gate before any search."""

import mlx.core as mx
import mlx.nn as nn


class Norm(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.w = mx.random.normal((dim,))

    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5) * self.w


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(23)
        self.norm = Norm()

    def __call__(self, x):
        return self.norm(x)


def build():
    return Model()
