"""Roofline (spec "Roofline"): a region's physical speed limit on this chip.

Bytes counts only what crosses the region's boundary, never the intermediate
round trips: those are the cost fusion hopes to delete, so they cannot be part
of the ideal. Flops are estimates from op and shape; they steer pricing and
the skip decision, nothing else.
"""

from __future__ import annotations

import math
from typing import Iterable

from ..measure.peaks import Peaks
from ..trace.types import Trace, TraceNode
from .build import is_view
from .types import Region, Roofline, Stretch

_DTYPE_BYTES = {
    "bool": 1, "uint8": 1, "int8": 1, "uint16": 2, "int16": 2, "float16": 2,
    "bfloat16": 2, "uint32": 4, "int32": 4, "float32": 4, "uint64": 8,
    "int64": 8, "complex64": 8,
}


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _bytes_of(spec: tuple[tuple[int, ...], str]) -> int:
    shape, dtype = spec
    return _numel(shape) * _DTYPE_BYTES.get(dtype, 4)


def node_flops(node: TraceNode) -> float:
    """Estimated flops from op and recorded shapes. A matmul is 2*M*N*K; a
    norm a few passes over N; elementwise one per output element."""
    out_elems = sum(_numel(s) for s, _ in node.out_specs)
    op = node.op
    if op in ("mx.matmul", "array.__matmul__", "mx.addmm", "mx.quantized_matmul",
              "mx.gather_qmm", "mx.block_masked_mm", "mx.gather_mm"):
        # K is the last dim of the first operand; output already holds M*N
        k = node.in_specs[0][0][-1] if node.in_specs[0][0] else 1
        return 2.0 * out_elems * k
    if op in ("mx.fast.rms_norm", "mx.fast.layer_norm"):
        return 4.0 * out_elems
    if op in ("mx.softmax", "mx.logsumexp"):
        return 5.0 * out_elems
    if op == "mx.fast.scaled_dot_product_attention":
        # q @ k^T and attn @ v: 2 * 2 * B*H*Lq*D*Lk
        q_shape = node.in_specs[0][0]
        k_shape = node.in_specs[1][0]
        lk = k_shape[-2] if len(k_shape) >= 2 else 1
        return 4.0 * _numel(q_shape) * lk
    if op in ("mx.fast.rope",):
        return 6.0 * out_elems
    if is_view(node):
        return 0.0
    if op in ("mx.sum", "mx.mean", "mx.max", "mx.min", "mx.prod", "mx.var", "mx.std"):
        return float(sum(_numel(s) for s, _ in node.in_specs))
    return float(out_elems)


def stretch_roofline(
    trace: Trace, stretch: Stretch, peaks: Peaks, t_orig_ms: float
) -> Roofline:
    nodes = trace.nodes[stretch.start_seq:stretch.end_seq + 1]
    spec_of = _spec_index(trace)

    boundary_bytes = sum(_bytes_of(spec_of[aid]) for aid in stretch.input_ids)
    boundary_bytes += sum(_bytes_of(spec_of[aid]) for aid in stretch.output_ids)
    flops = sum(node_flops(n) for n in nodes)
    launches = 1  # the ideal kernel for a region is one launch, whatever the library fires

    compute_dtype = _dominant_dtype(nodes)
    peak_flops = peaks.flops_gflops.get(compute_dtype)
    if peak_flops is None:
        peak_flops = max(peaks.flops_gflops.values()) if peaks.flops_gflops else 1e3

    t_mem = boundary_bytes / (peaks.bandwidth_gbps * 1e9) * 1e3
    t_compute = flops / (peak_flops * 1e9) * 1e3
    t_launch = launches * peaks.launch_us / 1e3
    t_roof = max(t_mem, t_compute, t_launch)
    bound = {t_mem: "memory", t_compute: "compute", t_launch: "launch"}[t_roof]
    return Roofline(
        t_mem_ms=t_mem,
        t_compute_ms=t_compute,
        t_launch_ms=t_launch,
        t_roofline_ms=t_roof,
        bound=bound,
        s_max=(t_orig_ms / t_roof) if t_roof > 0 else 1.0,
    )


def _spec_index(trace: Trace) -> dict[int, tuple[tuple[int, ...], str]]:
    specs: dict[int, tuple[tuple[int, ...], str]] = {}
    for node in trace.nodes:
        for aid, spec in zip(node.in_arrays, node.in_specs):
            specs.setdefault(aid, spec)
        for aid, spec in zip(node.out_arrays, node.out_specs):
            specs.setdefault(aid, spec)
    return specs


def _dominant_dtype(nodes: Iterable[TraceNode]) -> str:
    for node in nodes:
        for _, dtype in node.out_specs:
            if dtype in ("float16", "bfloat16", "float32"):
                return dtype
    return "float32"
