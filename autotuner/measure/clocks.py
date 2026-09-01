"""Clocks: the step clock and the paired interleaved comparison.

Law 4: never subtract two separately timed quantities. Every A/B number here
comes from one session, alternating ABBA so thermal drift is common-mode.
Pacing idles land only at block boundaries, followed by ramp-warm samples
(spike_10), so no timed sample ever starts on ramped-down clocks.
Sign convention (plan 5.13): positive delta means the candidate is faster.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

from .session import Session

# SE(median) for normal noise is ~1.2533 * sigma / sqrt(n). The ship margin's
# "3 sigma of the interleaved samples" reads on the uncertainty of the measured
# win, so sigma_ms below is that standard error, not the raw per-sample spread.
_MEDIAN_SE_FACTOR = 1.2533


@dataclass(frozen=True)
class StepClock:
    median_ms: float
    samples_ms: tuple[float, ...]
    warm_ms: tuple[float, ...]


@dataclass(frozen=True)
class PairedComparison:
    baseline_ms: tuple[float, ...]
    candidate_ms: tuple[float, ...]
    deltas_ms: tuple[float, ...]  # per pair, baseline - candidate: positive = faster
    median_baseline_ms: float
    median_delta_ms: float
    spread_ms: float  # raw stdev of the paired deltas
    sigma_ms: float   # standard error of the median delta; margins use 3x this
    n: int
    ratios: tuple[float, ...] = ()   # per pair, candidate / baseline
    median_ratio: float = 0.0        # the drift-immune quantity: report this
    stability: float = 0.0           # 0..1, how far the pairs agreed

    def wins_by(self, margin_ms: float) -> bool:
        return self.median_delta_ms > max(margin_ms, 3.0 * self.sigma_ms)

    def loses_by(self, margin_ms: float) -> bool:
        return -self.median_delta_ms > max(margin_ms, 3.0 * self.sigma_ms)


def _ratio_stats(base: list[float], cand: list[float]) -> tuple[tuple[float, ...], float, float]:
    """Per-pair ratios, their median, and a 0..1 agreement score. The score
    reads the ratios rather than the absolute times: on a drifting machine the
    absolutes are meant to move and the ratios are not, so their spread is the
    honest confidence in the comparison."""
    ratios = tuple(c / b for b, c in zip(base, cand) if b > 0)
    if not ratios:
        return (), 0.0, 0.0
    median = statistics.median(ratios)
    if len(ratios) < 4 or median <= 0:
        return ratios, median, 0.0
    q1, _, q3 = statistics.quantiles(ratios, n=4)
    # median / (median + IQR): 1 when the pairs agreed exactly, 0.5 when the
    # spread equals the value, never saturating, so a hopeless measurement and
    # a merely noisy one stay distinguishable.
    return ratios, median, median / (median + (q3 - q1))


def step_clock(session: Session, fn: Callable[[], object], reps: int = 9) -> StepClock:
    """The honest cost of one call of fn: warm until stable, then the median."""
    warm = session.warm_until_stable(fn)
    samples = []
    for _ in range(reps):
        session.fresh_chunk((fn,))
        samples.append(session.timed(fn))
    session.settle()
    clock = StepClock(
        median_ms=statistics.median(samples) * 1e3,
        samples_ms=tuple(t * 1e3 for t in samples),
        warm_ms=tuple(t * 1e3 for t in warm),
    )
    session.log("step_clock", median_ms=clock.median_ms, n=reps)
    return clock


def compare(
    session: Session,
    baseline_fn: Callable[[], object],
    candidate_fn: Callable[[], object],
    pairs: int = 16,
    warm_baseline: bool = True,
) -> PairedComparison:
    """Paired interleaved A/B. Each ABBA block yields two pairs with opposite
    order, so drift within a block cancels across the pair set. Both arms are
    warmed first (a caller that just warmed the baseline may say so); the
    baseline is measured here, now, never reused."""
    if pairs < 2 or pairs % 2:
        raise ValueError(f"pairs must be even and >= 2, got {pairs}")
    if warm_baseline:
        session.warm_until_stable(baseline_fn)
    session.warm_until_stable(candidate_fn)
    base: list[float] = []
    cand: list[float] = []
    for _ in range(pairs // 2):
        session.fresh_chunk((baseline_fn, candidate_fn))
        a1 = session.timed(baseline_fn)
        b1 = session.timed(candidate_fn)
        b2 = session.timed(candidate_fn)
        a2 = session.timed(baseline_fn)
        base += [a1, a2]
        cand += [b1, b2]
    session.settle()
    deltas = [a - b for a, b in zip(base, cand)]
    spread = statistics.stdev(deltas)
    ratios, median_ratio, stability = _ratio_stats(base, cand)
    result = PairedComparison(
        baseline_ms=tuple(t * 1e3 for t in base),
        candidate_ms=tuple(t * 1e3 for t in cand),
        deltas_ms=tuple(d * 1e3 for d in deltas),
        median_baseline_ms=statistics.median(base) * 1e3,
        median_delta_ms=statistics.median(deltas) * 1e3,
        spread_ms=spread * 1e3,
        sigma_ms=_MEDIAN_SE_FACTOR * spread / (len(deltas) ** 0.5) * 1e3,
        n=len(deltas),
        ratios=ratios,
        median_ratio=median_ratio,
        stability=stability,
    )
    session.log(
        "compare",
        median_baseline_ms=result.median_baseline_ms,
        median_delta_ms=result.median_delta_ms,
        median_ratio=result.median_ratio,
        stability=round(result.stability, 3),
        sigma_ms=result.sigma_ms,
        n=result.n,
    )
    return result
