"""Fixture: part of the forward runs under the model's own mx.compile, applied
after the tracer installed. The tracer witnessed the compile, so while armed
the plain body records as ordinary ops (the scope stays replayable) and while
disarmed the compiled version runs at full speed."""

import mlx.core as mx
import mlx.nn as nn


class PartCompiled(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(19)
        self.w = mx.random.normal((8, 8))
        self.fast = mx.compile(lambda a: mx.tanh(a) * 2.0)

    def __call__(self, x):
        y = x @ self.w
        z = self.fast(y)
        return z + 1.0


def build():
    return PartCompiled()
