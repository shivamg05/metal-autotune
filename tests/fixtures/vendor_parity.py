"""Fixture: one big matmul, already vendor-optimal. The loop must ship nothing
here, and the final e2e must read ~0."""

import mlx.core as mx
import mlx.nn as nn


class JustMatmul(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(22)
        self.w = mx.random.normal((512, 512)) * 0.04

    def __call__(self, x):
        return x @ self.w


def build():
    return JustMatmul()
