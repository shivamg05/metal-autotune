"""Roofline (spec "Roofline"): a region's physical speed limit on this chip.

Bytes counts only what crosses the region's boundary, never the intermediate
round trips: those are the cost fusion hopes to delete, so they cannot be part
of the ideal. Flops are estimates from op and shape; they steer pricing and
the skip decision, nothing else.

The bytes-and-launch part of the limit is measured, not computed, when the
caller has a probe clock: one launch that streams the boundary, timed beside
the region in the same paired window (measure/probe.py). A dependent kernel
pays its launch and its stream in series, and the peak from a 512 MB pass is
out of reach for a 2 MB weight, so the arithmetic overstated headroom by a
third at the sizes a decode step is made of. The flops term stays arithmetic
against the matmul peak.
"""

from __future__ import annotations

import math
from typing import Iterable

from ..measure.peaks import Peaks
from ..trace.types import STATE_PREFIX, Trace, TraceNode
from .build import VIEW_OPS, is_view
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
    trace: Trace, stretch: Stretch, peaks: Peaks, t_orig_ms: float,
    floor_ms: float | None = None,
) -> Roofline:
    nodes = trace.nodes[stretch.start_seq:stretch.end_seq + 1]
    spec_of = trace.span_specs(stretch.start_seq, stretch.end_seq)

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
    if floor_ms is None:
        t_roof = max(t_mem, t_compute, t_launch)
        bound = {t_mem: "memory", t_compute: "compute", t_launch: "launch"}[t_roof]
    else:
        t_roof = max(floor_ms, t_compute)
        bound = "compute" if t_compute > floor_ms else ("memory" if t_mem >= t_launch else "launch")
    return Roofline(
        t_mem_ms=t_mem,
        t_compute_ms=t_compute,
        t_launch_ms=t_launch,
        t_roofline_ms=t_roof,
        bound=bound,
        s_max=(t_orig_ms / t_roof) if t_roof > 0 else 1.0,
        t_floor_ms=floor_ms,
    )


def step_floor(trace: Trace, peaks: Peaks, step_ms: float) -> dict:
    """The whole step against its own physics: the bytes that must come from
    outside it (weights, state the model keeps, its inputs) plus its outputs,
    the flops of every recorded op, and the launches the library fires. Room
    is the part of the step that is not physics, the most any kernel work on
    this workload could ever take."""
    produced: set[int] = set()
    outside: dict[int, int] = {}
    flops, launches = 0.0, 0
    for node in trace.nodes:
        for aid, spec in zip(node.in_arrays, node.in_specs):
            if aid not in produced:
                outside.setdefault(aid, _bytes_of(spec))
        if node.op.startswith(STATE_PREFIX):
            # state the step reads back from memory, and the launches the
            # object's own method fired
            for aid, spec in zip(node.out_arrays, node.out_specs):
                outside.setdefault(aid, _bytes_of(spec))
            launches += sum(op not in VIEW_OPS and op != "array.__getitem__"
                            for op in node.scalar_args["receiver"]["inner_ops"])
        else:
            launches += not is_view(node)
        produced.update(node.out_arrays)
        flops += node_flops(node)
    specs = trace.span_specs(0, len(trace.nodes) - 1)
    total = sum(outside.values()) + sum(_bytes_of(specs[a]) for a in trace.step_outputs if a in specs)
    dtype = _dominant_dtype(trace.nodes)
    peak = peaks.flops_gflops.get(dtype) or (max(peaks.flops_gflops.values()) if peaks.flops_gflops else 1e3)
    t_mem = total / (peaks.bandwidth_gbps * 1e9) * 1e3
    t_compute = flops / (peak * 1e9) * 1e3
    floor = max(t_mem, t_compute)
    return {"bytes_mb": total / 1e6, "gflop": flops / 1e9, "launches": launches,
            "t_mem_ms": t_mem, "t_compute_ms": t_compute, "floor_ms": floor,
            "step_ms": step_ms, "room": (1.0 - floor / step_ms) if step_ms > 0 else None}


def _dominant_dtype(nodes: Iterable[TraceNode]) -> str:
    for node in nodes:
        for _, dtype in node.out_specs:
            if dtype in ("float16", "bfloat16", "float32"):
                return dtype
    return "float32"
