"""The comparison rules for gates 5 and 8 (plan section 7): assoc-preserving
tolerance compare floored by the library's own run-to-run wobble, the
non-finite pattern rule, worst-offender reporting, and the adversarial
value-regime generators. Tolerance values live with the caller; this module
only applies the numbers it is handed.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx

FLOAT_DTYPES = (mx.float16, mx.bfloat16, mx.float32)

# Regime constants (plan gate 5). Non-float inputs pass through unchanged in
# every regime: their values are semantics (indices, masks), not magnitudes.
SCALE_UP = 1e3
SCALE_DOWN = 1e-4
OUTLIER_VALUE = 1e4     # fits fp16 (max 65504)
OUTLIER_COUNT = 3
REGIMES = ("scaled_up", "scaled_down", "outliers", "zeros", "nonfinite")


@dataclass(frozen=True)
class CompareResult:
    passed: bool
    reason: str                    # "" | shape | dtype | nonfinite_pattern | tolerance
    max_excess: float              # worst |c - r| minus allowed; <= 0 on a pass
    index: tuple[int, ...] | None  # the worst offender (or first pattern violation)
    detail: str = ""


def nonfinite_pattern_ok(candidate: mx.array, reference: mx.array) -> mx.array:
    """Elementwise: the candidate is finite wherever the reference is finite,
    NaN where NaN, and the same signed inf where inf."""
    finite_ok = mx.isfinite(reference) & mx.isfinite(candidate)
    nan_ok = mx.isnan(reference) & mx.isnan(candidate)
    inf_ok = mx.isinf(reference) & (candidate == reference)
    return finite_ok | nan_ok | inf_ok


def compare(
    candidate: mx.array,
    reference: mx.array,
    rtol: float,
    atol: float,
    wobble_floor: float = 0.0,
) -> CompareResult:
    """Assoc-preserving compare: |c - r| <= max(atol + rtol * |r|, wobble_floor)
    elementwise, after the non-finite pattern rule. wobble_floor is the
    library's own run-to-run wobble, measured on the spot by the caller."""
    if tuple(candidate.shape) != tuple(reference.shape):
        return CompareResult(False, "shape", float("inf"), None,
                             f"{tuple(candidate.shape)} != {tuple(reference.shape)}")
    if candidate.dtype != reference.dtype:
        return CompareResult(False, "dtype", float("inf"), None,
                             f"{candidate.dtype} != {reference.dtype}")
    if candidate.size == 0:
        return CompareResult(True, "", 0.0, None)

    c = candidate.astype(mx.float32)
    r = reference.astype(mx.float32)
    ok = nonfinite_pattern_ok(c, r)
    if not mx.all(ok).item():
        viol = ~ok
        return CompareResult(
            False, "nonfinite_pattern", float("inf"), _first_index(viol),
            f"{int(mx.sum(viol).item())} elements break the non-finite pattern",
        )

    allowed = mx.maximum(atol + rtol * mx.abs(r), wobble_floor)
    # non-finite positions already matched the pattern exactly: zero excess
    excess = mx.where(mx.isfinite(r), mx.abs(c - r) - allowed, mx.zeros_like(r))
    worst = float(mx.max(excess).item())
    index = _unravel(int(mx.argmax(excess).item()), tuple(reference.shape))
    if worst > 0:
        over = int(mx.sum(excess > 0).item())
        return CompareResult(False, "tolerance", worst, index,
                             f"{over} elements over tolerance")
    return CompareResult(True, "", worst, index)


def max_abs_diff(a: mx.array, b: mx.array) -> float:
    """The wobble measurement: worst |a - b| between two runs, 0.0 where both
    share a non-finite pattern, inf when the patterns differ."""
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return float("inf")
    if a.size == 0:
        return 0.0
    af, bf = a.astype(mx.float32), b.astype(mx.float32)
    if not mx.all(nonfinite_pattern_ok(af, bf)).item():
        return float("inf")
    diff = mx.where(mx.isfinite(bf), mx.abs(af - bf), mx.zeros_like(bf))
    return float(mx.max(diff).item())


def value_regimes(inputs: Sequence[mx.array], seed: int,
                  weights: Sequence[bool] | None = None,
                  scale_up: float = SCALE_UP,
                  outlier: float = OUTLIER_VALUE) -> dict[str, list[mx.array]]:
    """Gate 5's adversarial regimes at the recorded shapes: scaled up,
    scaled down 1e-4, outlier-injected (a few elements at one large value),
    zeros, and planted inf/NaN lanes. Weight inputs are constants of the
    frozen model and pass through every regime untouched, as non-float inputs
    do. Deterministic for a given seed and input list."""
    rng = _random.Random(seed)
    out: dict[str, list[mx.array]] = {name: [] for name in REGIMES}
    for k, a in enumerate(inputs):
        if a.dtype not in FLOAT_DTYPES or a.size == 0 or (weights and weights[k]):
            for name in REGIMES:
                out[name].append(a)
            continue
        out["scaled_up"].append(a * scale_up)
        out["scaled_down"].append(a * SCALE_DOWN)
        out["outliers"].append(_with_outliers(a, rng, outlier))
        out["zeros"].append(mx.zeros_like(a))
        out["nonfinite"].append(_with_nonfinite_lanes(a, rng))
    return out


def _with_outliers(a: mx.array, rng: _random.Random, value: float) -> mx.array:
    k = min(OUTLIER_COUNT, a.size)
    positions = rng.sample(range(a.size), k)
    flat = mx.arange(a.size).reshape(a.shape)
    mask = mx.zeros(a.shape, dtype=mx.bool_)
    for p in positions:
        mask = mask | (flat == p)
    return mx.where(mask, mx.array(value, dtype=a.dtype), a)


def _with_nonfinite_lanes(a: mx.array, rng: _random.Random) -> mx.array:
    if a.ndim == 0:
        return mx.array(float("nan"), dtype=a.dtype)
    n0 = a.shape[0]
    i_inf = rng.randrange(n0)
    i_nan = i_inf if n0 == 1 else (i_inf + 1 + rng.randrange(n0 - 1)) % n0
    lane = mx.arange(n0).reshape((n0,) + (1,) * (a.ndim - 1))
    b = mx.where(lane == i_inf, mx.array(float("inf"), dtype=a.dtype), a)
    return mx.where(lane == i_nan, mx.array(float("nan"), dtype=a.dtype), b)


def _first_index(viol: mx.array) -> tuple[int, ...]:
    """Index of the first True in a violation mask, deterministically: score
    each position by how early it is, then argmax."""
    n = viol.size
    score = viol.reshape(-1).astype(mx.int64) * mx.arange(n, 0, -1)
    return _unravel(int(mx.argmax(score).item()), tuple(viol.shape))


def _unravel(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    idx: list[int] = []
    for dim in reversed(shape):
        flat, r = divmod(flat, dim)
        idx.append(r)
    return tuple(reversed(idx))
