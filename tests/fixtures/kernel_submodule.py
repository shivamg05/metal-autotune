"""Fixture: part of the forward runs in the model's own custom Metal kernel,
built once at import and kept at module level, the way mlx_lm keeps its
recurrent-scan kernels. The tracer can name it by import path: it records as
one opaque call, no region may include it, and a wrapper on the scope around
it can still replay the scope by calling that path."""

import mlx.core as mx
import mlx.nn as nn

scale_kernel = mx.fast.metal_kernel(
    name="fixture_scale",
    input_names=["a", "n"],
    output_names=["out"],
    source="uint i = thread_position_in_grid.x;\nout[i] = a[i] * static_cast<T>(n);\n",
)


def scale(y: mx.array, n: int) -> mx.array:
    """y * n through the kernel, launched the way mlx_lm launches its own:
    a Python int among the inputs, the launch derived from the shapes."""
    return scale_kernel(
        inputs=[y, n], template=[("T", y.dtype)], grid=(y.size, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[y.shape], output_dtypes=[y.dtype],
    )[0]


class PartKernel(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(19)
        self.w = mx.random.normal((8, 8))

    def __call__(self, x):
        y = x @ self.w
        z = scale(y, x.shape[0])
        return z + 1.0


class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.part = PartKernel()

    def __call__(self, x):
        return self.part(x)


def build():
    return Outer()
