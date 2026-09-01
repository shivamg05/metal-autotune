"""Fixture: a compiled callable the tracer never witnessed plain, standing in
for anything a library ships pre-compiled that the source substitution cannot
recover. It must record as one opaque node, and ops inside it can never be
regions. The fixture clears the proxy's plain path to simulate that history."""

import mlx.core as mx
import mlx.nn as nn


class PartOpaque(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(19)
        self.w = mx.random.normal((8, 8))
        self.fast = mx.compile(lambda a: mx.tanh(a) * 2.0)
        if hasattr(self.fast, "_plain"):
            self.fast._plain = None  # as if compiled before the tracer existed

    def __call__(self, x):
        y = x @ self.w
        z = self.fast(y)
        return z + 1.0


def build():
    return PartOpaque()
