"""Fixture: a basic multi-axis slice read. The first axis passes through whole
(a swept dim), the last two are sliced, so the output view carries a start
offset with dense strides. Element order is preserved, so the compare is exact."""

import mlx.core as mx
import mlx.nn as nn


class SliceRead(nn.Module):
    def __call__(self, x):
        return x[:, 1:3, 4:8]


def build():
    return SliceRead()
