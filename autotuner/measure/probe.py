"""The floor probe: one launch that streams a region's boundary bytes.

A region's physical limit used to be arithmetic: boundary bytes over the
chip's peak bandwidth, flops over the flops peak, one launch, whichever is
largest. Two things made that number wrong at the sizes a decode step is
made of. The bandwidth peak comes from a 512 MB pass, which a 2 MB weight
never reaches: a dependent kernel pays its launch and its stream in series,
not the larger of the two. And the region's own clock came from another
minute of the job, on a machine whose speed moves 15% within a run.

The probe measures the limit instead: one kernel launch that reads every
input byte and writes every output byte as fast as this chip streams, timed
in the same chained, paired loop as the region. The region's time over the
probe's is the headroom, and the machine's speed cancels because both were
measured in the same seconds. MLX's own matvec runs within 0 to 17% of this
probe on the 2 MB to 6 MB weights of a small decoder (spike 12).

Each thread reads 64 contiguous bytes per step with four loads in flight,
adjacent threads adjacent bytes; outputs are written the same way. Bytes past
the last whole 16-byte chunk of an array are skipped, at most 15 per array,
and an input under 8 elements is not read at all: mx.fast.metal_kernel binds
those in Metal's constant address space, which the vector loads cannot
alias (pinned in tests). Every array a probe sees is freshly materialized
(captured or synthesized), so its buffer starts at offset zero, which the
vector loads need.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import mlx.core as mx

Spec = tuple[tuple[int, ...], str]   # (shape, dtype name), as the trace records them

MLX_CONSTANT_BELOW = 8    # inputs with fewer elements land in constant memory
PROBE_THREADGROUP = 256
PROBE_MIN_THREADS = 2048
PROBE_MAX_THREADS = 65536
_BYTES_PER_THREAD = 128   # two 64-byte steps per thread fills the chip from 2 MB up (spike 12)


def dtype_of(name: str) -> mx.Dtype:
    return getattr(mx, "bool_" if name == "bool" else name)


def dtype_name(arr: mx.array) -> str:
    return str(arr.dtype).removeprefix("mlx.core.")


def array_specs(arrays: Sequence[mx.array]) -> list[Spec]:
    return [(tuple(a.shape), dtype_name(a)) for a in arrays]


def floor_from(comp, link_loop_ms: float, iters: int, region_ms: float) -> float:
    """The probe's per-pass time in the frame of region_ms. comp paired the
    probe loop (baseline) against the region loop (candidate); link_loop_ms
    is the chain alone, which both loops pay. Scaling region_ms by the paired
    ratio keeps region over floor equal to what one window measured, whatever
    the machine did between this compare and the one behind region_ms."""
    probe = comp.median_baseline_ms - link_loop_ms
    region = probe - comp.median_delta_ms
    if probe <= 0 or region <= 0:
        return max(probe / iters, 1e-6)  # noise beat a tiny region: the probe's own reading
    return max(region_ms * probe / region, 1e-6)


def spec_bytes(spec: Spec) -> int:
    shape, dtype = spec
    return math.prod(shape) * dtype_of(dtype).size


def _read(i: int, nbytes: int) -> str:
    m = nbytes // 16
    return (f"{{ const device packed_uint4* v = (const device packed_uint4*)inp{i}; uint j = tid;\n"
            f"  for (; 4u * j + 3u < {m}u; j += n) {{ a0 += uint4(v[4u * j]); a1 += uint4(v[4u * j + 1u]);"
            f" a2 += uint4(v[4u * j + 2u]); a3 += uint4(v[4u * j + 3u]); }}\n"
            f"  for (uint k = ({m}u / 4u) * 4u + tid; k < {m}u; k += n) a0 += uint4(v[k]); }}")


def _write(i: int, nbytes: int) -> str:
    m = nbytes // 16
    return (f"{{ device packed_uint4* v = (device packed_uint4*)out{i};\n"
            f"  for (uint j = tid; j < {m}u; j += n) v[j] = packed_uint4(acc, acc, acc, acc); }}")


def probe_source(in_specs: Sequence[Spec], out_bytes: Sequence[int]) -> str:
    """The kernel body: sum every input's bytes so no load can be dropped,
    then write that sum over every output."""
    lines = ["uint tid = thread_position_in_grid.x; uint n = threads_per_grid.x;",
             "uint4 a0 = uint4(0), a1 = uint4(0), a2 = uint4(0), a3 = uint4(0);"]
    lines += [_read(i, spec_bytes(spec)) for i, spec in enumerate(in_specs)
              if math.prod(spec[0]) >= MLX_CONSTANT_BELOW]
    lines.append("uint4 s = a0 + a1 + a2 + a3; uint acc = s.x + s.y + s.z + s.w;")
    lines += [_write(i, nb) for i, nb in enumerate(out_bytes)]
    return "\n".join(lines)


def probe_threads(total_bytes: int) -> int:
    groups = -(-total_bytes // (_BYTES_PER_THREAD * PROBE_THREADGROUP))
    return min(PROBE_MAX_THREADS, max(PROBE_MIN_THREADS, groups * PROBE_THREADGROUP))


def stream_probe(in_specs: Sequence[Spec], out_specs: Sequence[Spec]
                 ) -> Callable[[Sequence[mx.array]], list[mx.array]]:
    """A one-launch pass over arrays of in_specs, in order, that returns
    arrays of out_specs. Its values mean nothing; its time is the floor."""
    in_bytes = [spec_bytes(s) for s in in_specs]
    out_bytes = [spec_bytes(s) for s in out_specs]
    threads = probe_threads(sum(in_bytes) + sum(out_bytes))
    kernel = mx.fast.metal_kernel(
        name="stream_probe",
        input_names=[f"inp{i}" for i in range(len(in_specs))],
        output_names=[f"out{i}" for i in range(len(out_specs))],
        source=probe_source(in_specs, out_bytes),
    )
    shapes = [tuple(shape) for shape, _ in out_specs]
    dtypes = [dtype_of(d) for _, d in out_specs]

    def run(arrays: Sequence[mx.array]) -> list[mx.array]:
        return kernel(inputs=list(arrays), output_shapes=shapes, output_dtypes=dtypes,
                      grid=(threads, 1, 1), threadgroup=(PROBE_THREADGROUP, 1, 1))
    return run
