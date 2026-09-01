"""Fixture: four structurally identical layers. Copy grouping must price their
shared op sequence as one region with copies=4."""

import mlx.core as mx
import mlx.nn as nn


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = mx.ones((16,))
        self.w = mx.random.normal((16, 16))

    def __call__(self, x):
        h = mx.fast.rms_norm(x, self.g, eps=1e-5)
        return x + h @ self.w


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(20)
        self.layers = [Layer() for _ in range(4)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build():
    return Stack()
