"""Fixture: the composed case. Split the last axis into two views, run a
different elementwise op on each, then concatenate them back. It proves offset
views feed later stages and that concat reads its sources through their views.
The second split part starts at a nonzero offset, so both offset paths run."""

import mlx.core as mx
import mlx.nn as nn


class ConcatJoin(nn.Module):
    def __call__(self, x):
        a, b = mx.split(x, 2, axis=-1)
        return mx.concatenate([a * 2.0, b + 1.0], axis=-1)


def build():
    return ConcatJoin()
