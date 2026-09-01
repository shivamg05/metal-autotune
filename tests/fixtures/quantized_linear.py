"""Fixture: two quantized projections with elementwise glue, the shape of
every quantized checkpoint's hot path. First 4-bit then 8-bit, both affine
with transposed weights, so one lowered kernel covers both unpack widths."""

import mlx.core as mx
import mlx.nn as nn


class QuantizedChain(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(23)
        w4 = mx.random.normal((64, 128), dtype=mx.float16)
        self.wq4, self.scales4, self.biases4 = mx.quantize(w4, group_size=32, bits=4)
        w8 = mx.random.normal((32, 64), dtype=mx.float16)
        self.wq8, self.scales8, self.biases8 = mx.quantize(w8, group_size=32, bits=8)

    def __call__(self, x):
        h = mx.quantized_matmul(x, self.wq4, scales=self.scales4, biases=self.biases4,
                                transpose=True, group_size=32, bits=4)
        h = h * 0.5
        return mx.quantized_matmul(h, self.wq8, scales=self.scales8, biases=self.biases8,
                                   transpose=True, group_size=32, bits=8)


def build():
    return QuantizedChain()
