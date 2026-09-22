"""Fixture: scores scaled by a constant built inside forward,
mx.sqrt(mx.array(float(d))). mx.array is the array class and cannot be
wrapped, so the recorder records the small constant as a creation call; held
to "every array needs a recorded producer" without that, the trace aborted
(three MetalBench attention problems died this way)."""

import mlx.core as mx
import mlx.nn as nn


class Scores(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.w = mx.random.normal((dim, dim))

    def __call__(self, x):
        d = x.shape[-1]
        return mx.maximum((x @ self.w) / mx.sqrt(mx.array(float(d))), 0.0) * 0.5


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(29)
        self.scores = Scores()

    def __call__(self, x):
        return self.scores(x)


def build():
    return Model()
