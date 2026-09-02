"""Executable controls: the A/A null and the bit-identical slowdown.

The A/A null must not ship; the planted slowdown must be detected. Both run
against the same public clocks the real harness uses, and the A/A null runs at
job start to calibrate and log the session's floor.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx

from .clocks import PairedComparison, compare
from .session import Session

SYNTH_N = 1024
SYNTH_DEPTH = 12
_RESCALE = 0.03125  # exact power of two: kills growth without rounding


def synthetic_workload(
    depth: int = SYNTH_DEPTH,
    n: int = SYNTH_N,
    injected_links: int = 0,
    seed: int = 0,
) -> Callable[[], mx.array]:
    """A dependent matmul chain producing tens of ms of GPU work per call.

    injected_links > 0 plants the bit-identical slowdown: those links compute
    the same matmul twice and average the two results. The average equals
    either copy bitwise (deterministic kernel, exact doubling and halving), so
    outputs never change while the work provably grows by injected_links/depth.
    """
    if not 0 <= injected_links <= depth:
        raise ValueError(f"injected_links must be in [0, {depth}]")
    keys = mx.random.split(mx.random.key(seed), depth + 1)
    x = mx.random.normal((n, n), key=keys[0])
    weights = [mx.random.normal((n, n), key=keys[i + 1]) for i in range(depth)]
    mx.eval(x, weights)

    def fn() -> mx.array:
        y = x
        for i, w in enumerate(weights):
            if i < injected_links:
                a = (y @ w) * _RESCALE
                b = (y @ w) * _RESCALE
                y = (a + b) * 0.5
            else:
                y = (y @ w) * _RESCALE
        return y

    return fn


def aa_null(
    session: Session,
    fn: Callable[[], object] | None = None,
    pairs: int = 16,
) -> PairedComparison:
    """Same callable both arms. The result's sigma is the session floor; a
    significant 'win' here means the session cannot be trusted."""
    fn = fn or synthetic_workload()
    result = compare(session, fn, fn, pairs=pairs)
    session.log(
        "aa_floor",
        median_delta_ms=result.median_delta_ms,
        sigma_ms=result.sigma_ms,
        floor_pct=round(3 * result.sigma_ms / result.median_baseline_ms * 100, 3),
    )
    return result
