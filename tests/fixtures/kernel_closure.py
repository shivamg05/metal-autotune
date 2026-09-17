"""Fixture: a custom kernel built inside the model and held on it. Nothing
outside the model can reach it by import path, so it records as an unnamed
opaque call, no region may include it, and no wrapper can replay a scope that
contains it."""

import mlx.core as mx
import mlx.nn as nn


class PartKernelUnnamed(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(19)
        self.w = mx.random.normal((8, 8))
        self.double = mx.fast.metal_kernel(
            name="fixture_double_held",
            input_names=["a"],
            output_names=["out"],
            source="uint i = thread_position_in_grid.x;\nout[i] = a[i] * static_cast<T>(2);\n",
        )

    def __call__(self, x):
        y = x @ self.w
        z = self.double(
            inputs=[y], template=[("T", y.dtype)], grid=(y.size, 1, 1), threadgroup=(32, 1, 1),
            output_shapes=[y.shape], output_dtypes=[y.dtype],
        )[0]
        return z + 1.0


def build():
    return PartKernelUnnamed()
