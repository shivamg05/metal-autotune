"""Pool saturation: the MLX allocator recycles buffers without
zeroing, so a computation that reads recycled memory can see stale, plausible
values. Saturating the pool with NaN-filled buffers of the sizes about to be
allocated makes that read NaN instead. init_value closes the output-buffer
hole on candidate launches; this is the belt and braces around library
reference computations.
"""

from __future__ import annotations

import math
from typing import Iterable

import mlx.core as mx

DEFAULT_COPIES = 4


def saturate_pool(sizes_bytes: Iterable[int], copies: int = DEFAULT_COPIES) -> int:
    """Allocate `copies` NaN-filled buffers per byte size, eval, free them back
    to the pool. Sizes are byte sizes (padded up to float32 elements) so one
    call covers outputs of any dtype. Returns the bytes dirtied."""
    buffers = []
    total = 0
    for nbytes in sizes_bytes:
        n = max(1, math.ceil(nbytes / 4))
        for _ in range(copies):
            buffers.append(mx.full((n,), float("nan"), dtype=mx.float32))
            total += n * 4
    mx.eval(buffers)
    del buffers  # freed buffers return to the pool still NaN-filled
    return total
