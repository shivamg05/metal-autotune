"""Fixture: an elementwise op then an equal-sections split of the last axis.
Each part is a view of the same buffer at a different start offset; both are
step outputs, so the region has two outputs. Order is preserved, compare exact."""

import mlx.core as mx
import mlx.nn as nn


class SplitHeads(nn.Module):
    def __call__(self, x):
        return mx.split(mx.abs(x), 2, axis=-1)


def build():
    return SplitHeads()
