"""Clocks: the step clock and the paired interleaved comparison.

Law 4: never subtract two separately timed quantities. Every A/B number here
comes from one session, alternating ABBA so thermal drift is common-mode.
Pacing idles land only at block boundaries, followed by ramp-warm samples,
so no timed sample ever starts on ramped-down clocks. Sign convention:
positive delta means the candidate is faster.
"""

from __future__ import annotations

import statistics

import mlx.core as mx
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from .session import Session

# SE(median) for normal noise is ~1.2533 * sigma / sqrt(n). The ship margin's
# "3 sigma of the interleaved samples" reads on the uncertainty of the measured
# win, so sigma_ms below is that standard error, not the raw per-sample spread.
_MEDIAN_SE_FACTOR = 1.2533

# One timed sample of a looped region holds this much work, so the fixed
# submit-and-sync cost of an evaluation is a rounding error in the per-pass
# figure. The pricing clock and the ship clock size their loops with the
# same rule, or every headroom figure would carry the gap between them.
CLOCK_TARGET_MS = 20.0
CLOCK_EST_ITERS = 10   # passes in the estimate that sizes the loop
CLOCK_MIN_ITERS = 20   # never fewer passes per sample, however slow the pass
CLOCK_MAX_ITERS = 2000


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


# A timed loop must read its bytes from memory and run its passes one after
# another, the way a model step does. Rotating a working set this large keeps
# the data out of the GPU's cache; chaining the passes keeps Metal from running
# independent launches side by side, which makes a one-threadgroup kernel look
# many times faster than it is in a model, where each layer waits for the last.
CACHE_DEFEAT_BYTES = 128 * 1024 * 1024
MAX_TIMING_SETS = 16


def synthesize_like(arrays: Mapping[int, mx.array], seed: int) -> dict[int, mx.array]:
    """Timing-only inputs: the same shapes and dtypes, random values. Never
    used for a correctness comparison."""
    keys = mx.random.split(mx.random.key(seed), max(len(arrays), 1))
    out = {}
    for i, (aid, arr) in enumerate(sorted(arrays.items())):
        if arr.dtype in (mx.float16, mx.bfloat16, mx.float32):
            out[aid] = mx.random.normal(arr.shape, dtype=arr.dtype, key=keys[i])
        else:
            out[aid] = arr  # integer inputs (indices) keep real values
    mx.eval(list(out.values()))
    return out


def timing_sets(sets: Sequence[Mapping[int, mx.array]]) -> list[dict[int, mx.array]]:
    """The sets a timed loop rotates over: the given ones plus synthesized
    look-alikes until the working set is too big for the cache."""
    out = [dict(s) for s in sets]
    set_bytes = sum(a.nbytes for a in out[0].values())
    seed = 0
    while set_bytes * len(out) < CACHE_DEFEAT_BYTES and len(out) < MAX_TIMING_SETS:
        out.append({**out[0], **synthesize_like(out[0], 7000 + seed)})
        seed += 1
    return out


def link_input(binds: Mapping[int, mx.array], weight_ids: Iterable[int] = ()) -> int | None:
    """The input the chain link rides on: the smallest float input that is not
    a weight (a weight would cost a copy per pass), else the smallest float
    input; None when nothing float enters, which leaves the loop unchained."""
    floats = {a: arr for a, arr in binds.items() if arr.dtype in (mx.float16, mx.bfloat16, mx.float32)}
    pool = {a: arr for a, arr in floats.items() if a not in set(weight_ids)} or floats
    return min(pool, key=lambda a: pool[a].nbytes) if pool else None


def chained_loop(pass_fn: Callable[[Mapping[int, mx.array]], object],
                 sets: Sequence[Mapping[int, mx.array]], iters: int,
                 link_id: int | None) -> Callable[[], list]:
    """iters passes over the rotating sets, each pass's linked input carrying
    a zero taken from the previous pass's first output, so no pass can start
    before the last one ends. The link costs the same on every arm; paired
    against link_loop, it drops out of a per-pass figure."""
    def loop() -> list:
        outs, link = [], None
        for i in range(iters):
            binds = sets[i % len(sets)]
            if link is not None:
                binds = {**binds, link_id: binds[link_id] + link}
            out = pass_fn(binds)
            outs.append(out)
            if link_id is not None:
                first = out[0] if isinstance(out, (list, tuple)) else out
                link = (first.reshape(-1)[0] * 0).astype(binds[link_id].dtype)
        return outs
    return loop


def link_loop(sets: Sequence[Mapping[int, mx.array]], iters: int, link_id: int) -> Callable[[], list]:
    """The chain alone: the same loop around a pass that only hands its
    linked input back. Paired against a real loop, it takes the link's cost
    and a sample's fixed submit-and-sync cost out of the per-pass figure."""
    return chained_loop(lambda b: [b[link_id]], sets, iters, link_id)


def loop_iterations(timer: Callable[[Callable[[], object]], float],
                    loop_for: Callable[[int], Callable[[], object]],
                    target_ms: float = CLOCK_TARGET_MS) -> int:
    """How many passes one timed sample should hold. loop_for(n) is the loop
    of n passes as it will be timed. The per-pass estimate is the difference
    between a longer and a shorter warm loop, so a sample's fixed
    submit-and-sync cost, which a busy GPU can push to several milliseconds,
    is not mistaken for pass time. The first loop is thrown away (Metal
    compile, post-idle clock ramp)."""
    short, long = CLOCK_EST_ITERS, 4 * CLOCK_EST_ITERS
    timer(loop_for(short))
    t_short = timer(loop_for(short))
    t_long = timer(loop_for(long))
    t_est_ms = (t_long - t_short) / (long - short) * 1e3
    return max(CLOCK_MIN_ITERS, min(int(target_ms / max(t_est_ms, 1e-3)), CLOCK_MAX_ITERS))


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
