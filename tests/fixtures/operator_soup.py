"""Fixture: dunder coverage. Plain add, reflected sub/mul, in-place add, matmul,
slice read, slice write, comparison, unary neg. The tracer must record every one."""

import mlx.core as mx
import mlx.nn as nn


class Soup(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(11)
        self.w = mx.random.normal((8, 8))

    def __call__(self, x):
        y = x + 1.0
        y = 2.0 * y
        y = 1.0 - y
        y += x
        z = y @ self.w
        row = z[1]
        z = z * (row > 0)
        buf = mx.zeros_like(z)
        buf[0] = -row
        return z + buf


def build():
    return Soup()
