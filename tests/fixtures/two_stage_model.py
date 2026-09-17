"""Two ordinary operations in an installable child module."""
import mlx.nn as nn


class Step(nn.Module):
    def __call__(self, x):
        return x * 2.0 + 1.0


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = Step()

    def __call__(self, x):
        return self.step(x)


def build():
    return Model()
