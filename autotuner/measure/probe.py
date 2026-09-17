"""A generic ranking probe: one launch streams a region's boundary arrays.

Cost varies with bytes, array sizes, launch overhead and cache/device conditions.
The probe is paired with the region in the same measurement window to reduce
clock drift. It estimates removable data movement, not a guaranteed minimum:
its own implementation has overhead and it does not perform the region's math.
It applies equally to library operations and captured custom kernels.

Vector loads omit tails shorter than 16 bytes and inputs under 8 elements,
which MLX binds in Metal's constant address space. Captured or synthesized
arrays are materialized at buffer offset zero for vector access.
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
COMPUTE_PROBE_N = 4096    # a plain matmul this square reaches the chip's best rate; 2048 reads 5% under it, 1024 a quarter


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


def compute_probe(dtype_name: str, n: int = COMPUTE_PROBE_N) -> tuple[Callable[[], mx.array], float]:
    """The arithmetic ceiling's probe: a plain n-square matmul at this dtype,
    the fastest the chip does math, timed beside the regions it bounds.
    Returns the pass and its flops."""
    dtype = dtype_of(dtype_name)
    a = mx.random.normal((n, n)).astype(dtype)
    b = mx.random.normal((n, n)).astype(dtype)
    mx.eval(a, b)
    return (lambda: a @ b), 2.0 * n ** 3


def stream_probe(in_specs: Sequence[Spec], out_specs: Sequence[Spec]
                 ) -> Callable[[Sequence[mx.array]], list[mx.array]]:
    """A one-launch pass over arrays of in_specs, in order, that returns
    arrays of out_specs. Its values mean nothing; its time is a ranking estimate."""
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
