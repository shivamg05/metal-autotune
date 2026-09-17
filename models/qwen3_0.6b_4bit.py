"""Qwen3 0.6B Base at 4 bits, exactly as mlx_lm loads it, quantized in memory.

The bf16 checkpoint is quantized after loading (4 bits, groups of 64, mlx_lm's
own recipe), so no second checkpoint downloads. At 4 bits a decode step streams
335 MB of weights instead of 1.19 GB, so launches and glue, the part kernel
work can take, are a far larger share of the step. The projections record as
mx.quantized_matmul, the one op whose starting kernel the harness stitches from
MLX's own Metal source.

The manifest decides what one call is: a prompt (`shape: [1, L]`) or one token
over a conversation already in place (`shape: [1, 1]` plus `context: 512`).
The weights download from Hugging Face on the first build (about 1.2 GB).
"""

import mlx.nn as nn

MODEL = "Qwen/Qwen3-0.6B-Base"
BITS = 4
GROUP_SIZE = 64


def build():
    from mlx_lm import load

    model, _ = load(MODEL)
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    return model
