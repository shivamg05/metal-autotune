"""Shared numeric helpers and adversarial input regimes for region validation."""

from __future__ import annotations

import random as _random
from typing import Sequence

import mlx.core as mx

from autotuner_runtime.numeric import (FLOAT_DTYPES, CompareResult, compare,
                                        nonfinite_pattern_ok)

# Regime constants for gate 5. Non-float inputs pass through unchanged in
# every regime: their values are semantics (indices, masks), not magnitudes.
SCALE_UP = 1e3
SCALE_DOWN = 1e-4
OUTLIER_VALUE = 1e4     # fits fp16 (max 65504)
OUTLIER_COUNT = 3
REGIMES = ("scaled_up", "scaled_down", "outliers", "zeros", "nonfinite")



def max_abs_diff(a: mx.array, b: mx.array) -> float:
    """The wobble measurement: worst |a - b| between two runs, 0.0 where both
    share a non-finite pattern, inf when the patterns differ."""
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return float("inf")
    if a.size == 0:
        return 0.0
    if a.dtype not in FLOAT_DTYPES:
        return 0.0 if mx.array_equal(a, b).item() else float("inf")
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
