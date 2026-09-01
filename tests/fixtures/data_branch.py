"""Fixture: data-dependent control flow. The trace holds only the taken path;
paths the workload never took are the sweep's problem, not the ranking's."""

import mlx.core as mx
import mlx.nn as nn


class Branchy(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(18)
        self.w = mx.random.normal((8, 8))

    def __call__(self, x):
        y = x @ self.w
        if y.sum().item() > 0:
            return mx.tanh(y)
        return mx.sigmoid(y)


def build():
    return Branchy()
