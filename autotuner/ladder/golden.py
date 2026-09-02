"""The fp32 golden evaluator for assoc-changing compares (gate 8).

The golden is the region's own recorded ops replayed with float bindings
promoted to fp32 and quantized ops routed through a substitution table
(dequantize + matmul in fp32, group_size/bits pinned to the recorded scalar
args). Candidates and the library are both scored against it:
err(candidate) <= kappa * err(library) + floor, kappa and floor caller-held.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import mlx.core as mx

from autotuner.trace.replay import replay
from autotuner.trace.types import TraceNode

from .numeric import FLOAT_DTYPES, nonfinite_pattern_ok

DEFAULT_KAPPA = 1.25
DEFAULT_DENOM_CLAMP = 1e-6


def _quantized_matmul_fp32(x, w, scales, biases=None, transpose=True,
                           group_size=None, bits=None, mode="affine", *,
                           stream=None):
    """mx.quantized_matmul -> dequantize + matmul in fp32. group_size, bits,
    and mode flow through exactly as recorded; a recorded None hits the same
    library defaults quantized_matmul itself resolves (verified: the defaults
    of quantize/dequantize/quantized_matmul agree on this mlx version)."""
    scales = scales.astype(mx.float32)
    if biases is not None:
        biases = biases.astype(mx.float32)
    wf = mx.dequantize(w, scales, biases, group_size=group_size, bits=bits, mode=mode)
    xf = x.astype(mx.float32)
    return mx.matmul(xf, mx.swapaxes(wf, -1, -2) if transpose else wf)


_DEFAULT_TABLE: dict[str, Callable] = {
    "mx.quantized_matmul": _quantized_matmul_fp32,
}


def substitution_table(extra: Mapping[str, Callable] | None = None) -> dict[str, Callable]:
    """The quantized-op substitution table, extensible as traced ops require."""
    table = dict(_DEFAULT_TABLE)
    if extra:
        table.update(extra)
    return table


def promote_bindings(bindings: Mapping[int, mx.array]) -> dict[int, mx.array]:
    """Float bindings to fp32 (exact for fp16/bf16); everything else, including
    packed quantized weights, untouched."""
    return {
        aid: a.astype(mx.float32) if a.dtype in FLOAT_DTYPES and a.dtype != mx.float32 else a
        for aid, a in bindings.items()
    }


def golden_outputs(
    nodes: Sequence[TraceNode],
    bindings: Mapping[int, mx.array],
    outputs: Iterable[int],
    substitutions: Mapping[str, Callable] | None = None,
) -> dict[int, mx.array]:
    """Replay the span with promoted bindings and the substitution table."""
    table = substitution_table() if substitutions is None else dict(substitutions)
    return replay(nodes, promote_bindings(bindings), outputs, op_substitute=table)


def err(
    candidates: Sequence[mx.array],
    goldens: Sequence[mx.array],
    denom_clamp: float = DEFAULT_DENOM_CLAMP,
) -> float:
    """Worst per-output max relative error against the golden, denominator
    clamped. Where the golden is non-finite: 0 if the candidate reproduces the
    pattern, inf if not."""
    worst = 0.0
    for c, g in zip(candidates, goldens, strict=True):
        if tuple(c.shape) != tuple(g.shape):
            raise ValueError(f"output shape {tuple(c.shape)} != golden {tuple(g.shape)}")
        if c.size == 0:
            continue
        cf, gf = c.astype(mx.float32), g.astype(mx.float32)
        ok = nonfinite_pattern_ok(cf, gf)
        rel = mx.abs(cf - gf) / mx.maximum(mx.abs(gf), denom_clamp)
        rel = mx.where(
            mx.isfinite(gf) & ok, rel,
            mx.where(ok, mx.zeros_like(rel), mx.array(float("inf"))),
        )
        worst = max(worst, float(mx.max(rel).item()))
    return worst


def passes(candidate_err: float, library_err: float,
           kappa: float = DEFAULT_KAPPA, floor: float = 0.0) -> bool:
    """The assoc-changing gate: reordering accumulation is allowed, being
    sloppier than the library is not."""
    return candidate_err <= kappa * library_err + floor
