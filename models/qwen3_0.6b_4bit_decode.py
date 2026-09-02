"""Qwen3 0.6B Base at 4 bits, as one repeatable decode step.

The same model file as qwen3_0.6b_decode.py, quantized in memory after
loading (4 bits, groups of 64, mlx_lm's own recipe), so nothing more is
downloaded. The point is what it does to the step: bf16 decode reads 1.19 GB
of weights and spends 92% of its 14.4 ms on that stream, which no kernel can
shorten; at 4 bits the stream is 335 MB and 4.2 ms of a 5.7 ms step, so a
quarter of the step is launches and glue, the part kernel work can take.
The projections record as mx.quantized_matmul, the one op whose starting
kernel the harness stitches from MLX's own Metal source.
"""

import importlib.util
from pathlib import Path

BITS = 4
GROUP_SIZE = 64

_spec = importlib.util.spec_from_file_location(
    "qwen3_0_6b_decode", Path(__file__).with_name("qwen3_0.6b_decode.py"))
_bf16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bf16)


def build():
    return _bf16.build_step(bits=BITS, group_size=GROUP_SIZE)
