"""The directions a region's widening round opens from (spec "Widening
round"). Hardware facts only: each names a structure, what it trades, and
the bounds it can pay under. The region's bound masks the rest, the job seed
fixes the order, and the judge may add a direction of its own.
"""

from __future__ import annotations

import random

OPENERS = 4  # attempts a widening round holds at most; fewer when fewer directions are offered

# kind, the direction in one line, the bounds it can pay under
DIRECTIONS = (
    ("one-dispatch", "the whole region as one dispatch, one thread per output element, "
     "intermediates in registers and re-rounded at each recorded op boundary; no synchronization",
     ("memory", "launch")),
    ("thread-per-row", "one thread per row looping over its columns; no synchronization; "
     "pays when rows are short", ("memory",)),
    ("simdgroup-per-row", "one SIMD group per output row, lanes stride the reduction, simd_sum "
     "at the end; no barriers or threadgroup memory; pays on dot products with few rows",
     ("memory", "compute")),
    ("threadgroup-per-row", "one threadgroup per output row, the reduction split across its SIMD "
     "groups and combined in threadgroup memory; more groups in flight when rows are long and few",
     ("memory", "compute")),
    ("split-k", "the reduction split across threadgroups with a second stage combining the "
     "partials; changes the summation order, so tag it changing", ("memory", "compute")),
    ("2d-tile", "a threadgroup owns a tile of rows and columns and loads its operands "
     "cooperatively; pays on matmul shapes", ("memory", "compute")),
    ("persistent", "fewer threadgroups than tiles, each looping over tiles; amortizes the "
     "per-group prologue when tiles are many and small", ("memory", "compute")),
    ("wide-loads", "16-byte loads and stores by reinterpreting the element type; fewer load "
     "instructions per byte", ("memory",)),
    ("register-preload", "issue every load first, then compute from registers; more loads in "
     "flight per thread", ("memory",)),
    ("threadgroup-staging", "an operand many threads read is loaded once per threadgroup into "
     "threadgroup memory; costs a barrier", ("memory", "compute")),
    ("output-reuse", "one thread produces several outputs that share an input, loading it "
     "once; costs registers", ("memory", "compute")),
    ("recompute", "a cheap intermediate is recomputed where it is used instead of stored and "
     "read back", ("memory",)),
    ("half-exact-mul", "native half multiplies where the product is exact in float, so the "
     "rounding matches; never for sums or exp", ("compute",)),
    ("simdgroup-matrix", "the SIMD group matrix unit (simdgroup_multiply_accumulate) for matmul "
     "shapes; changes the accumulation order, so tag it changing", ("compute",)),
    ("reduction-tree", "a fixed tree order for a reduction instead of a serial loop; changes "
     "the summation order, so tag it changing", ("compute",)),
    ("unroll", "loop bounds as compile-time constants so the loop unrolls and its counters "
     "vanish; needs a fallback predicate outside those shapes", ("compute", "launch")),
    ("compile-time-shapes", "shapes as compile-time constants with a fallback predicate, no "
     "shape buffers bound; shorter encode and prologue", ("launch",)),
    ("grid-shape", "group size and grid dimensions matched to the core count, several groups "
     "per core; a 1D grid against a 2D one", ("launch", "memory")),
    ("fewer-buffers", "fewer bound buffers: unused shape and stride buffers dropped, scalars "
     "packed; shorter encode", ("launch",)),
)
assert len({kind for kind, _, _ in DIRECTIONS}) == len(DIRECTIONS)


def directions_for(bound: str | None, seed: int, fingerprint: str) -> list[dict]:
    """The directions offered for one region: those that can pay under its
    bound, all of them when the bound is unknown, in an order the job seed
    and the region fix."""
    offered = [{"kind": kind, "direction": text, "pays": list(pays)}
               for kind, text, pays in DIRECTIONS if bound is None or bound in pays]
    random.Random(f"{seed}:{fingerprint}").shuffle(offered)
    return offered
