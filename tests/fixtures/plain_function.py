"""Fixture: a model that is a bare function, no nn.Module anywhere. Its ops must
carry the top-level address so a wrapper still has an install point."""

import mlx.core as mx

mx.random.seed(12)
_W = mx.random.normal((16, 16))


def build():
    def model(x):
        y = mx.maximum(x, 0.0)
        return y @ _W + 1.0

    return model
