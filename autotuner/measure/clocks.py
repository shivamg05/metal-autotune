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

from autotuner_runtime.stats import (_MEDIAN_SE_FACTOR, PairedComparison, _ratio_stats,
                                     comparison_from_samples, paired_means)

from .session import Session

# One timed sample of a looped region holds this much work, so the fixed
# submit-and-sync cost of an evaluation is a rounding error in the per-pass
# figure. The pricing clock and the ship clock size their loops with the
# same rule, or every headroom figure would carry the gap between them.
CLOCK_TARGET_MS = 20.0
CLOCK_EST_ITERS = 10   # passes in the estimate that sizes the loop
CLOCK_MIN_ITERS = 1    # a slow pass already amortizes submission cost
CLOCK_MAX_ITERS = 2000


@dataclass(frozen=True)
class StepClock:
    median_ms: float
    samples_ms: tuple[float, ...]
    warm_ms: tuple[float, ...]


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
                if isinstance(out, (list, tuple)) and not out:
                    raise ValueError("a timed pass returned no outputs; the chain has nothing to link")
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
    timer(loop_for(1))
    short = 1
    t_short = timer(loop_for(short))
    t_long = timer(loop_for(4 * short))
    if (t_long - t_short) * 1e3 < target_ms / 4:
        short = CLOCK_EST_ITERS
        t_short = timer(loop_for(short))
        t_long = timer(loop_for(4 * short))
    long = 4 * short
    t_est_ms = (t_long - t_short) / (long - short) * 1e3
    return max(CLOCK_MIN_ITERS, min(int(target_ms / max(t_est_ms, 1e-3)), CLOCK_MAX_ITERS))


def sample_group(session: Session, arms: dict[str, Callable[[], object]],
                 pairs: int = 8, *, defer_cooling: bool = False) -> dict[str, tuple[float, ...]]:
    """Measure several arms together, reversing their order every pass.

    The model arm is shared by all region prices in this group. A forward
    pass followed by its reverse brackets every other arm with that model,
    without rerunning a slow model separately for every candidate. Returns
    aligned sample rows in milliseconds; no old baseline enters a row.
    Deferred cooling requires the caller to gate its next GPU phase, or to
    hand the deadline to a parent that will do so.
    """
    if not arms or pairs < 2 or pairs % 2:
        raise ValueError("a group needs arms and an even number of pairs >= 2")
    names = list(arms)
    session.fresh_chunk(arms[names[0]])
    for name in names:
        session.warm_until_stable(arms[name])
    samples: dict[str, list[float]] = {name: [] for name in names}
    try:
        for index in range(pairs):
            for name in names if index % 2 == 0 else reversed(names):
                samples[name].append(session.timed(arms[name]) * 1e3)
    finally:
        if defer_cooling:
            session.defer_settle()
        else:
            session.settle()
    session.log("sample_group", arms=len(arms), pairs=pairs)
    return {name: tuple(values) for name, values in samples.items()}


def step_clock(session: Session, fn: Callable[[], object], reps: int = 9) -> StepClock:
    """The honest cost of one call of fn: warm until stable, then the median."""
    session.fresh_chunk(fn)
    warm = session.warm_until_stable(fn)
    samples = [session.timed(fn) for _ in range(reps)]
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
    defer_cooling: bool = False,
) -> PairedComparison:
    """Paired interleaved A/B. Alternate ABBA and BAAB blocks, averaging
    each arm's two samples before comparing them. Both surround the same midpoint,
    so linear drift cancels within each observation. A block is one
    observation, not two independent pairs. Both arms are warmed first (a
    caller that just warmed the baseline may say so); the baseline is
    measured here, now, never reused.

    Pacing happens once, before the warms, so every sample of the comparison
    is taken in the state the model actually runs in. Idling between samples
    of one comparison measured a machine nobody deploys on: the same step read
    5.7 ms back to back and 8-16 ms paced."""
    if pairs < 2 or pairs % 2:
        raise ValueError(f"pairs must be even and >= 2, got {pairs}")
    warmed = session.fresh_chunk(baseline_fn)
    if warm_baseline and not warmed:
        session.warm_until_stable(baseline_fn)
    session.warm_until_stable(candidate_fn)
    base: list[float] = []
    cand: list[float] = []
    for block in range(pairs // 2):
        # Balance which arm occupies the outside slots. Fixed ABBA cancels
        # linear drift but gives a recurring slot/cache bias to the same arm.
        order = ("a", "b", "b", "a") if block % 2 == 0 else ("b", "a", "a", "b")
        for arm in order:
            if arm == "a":
                base.append(session.timed(baseline_fn))
            else:
                cand.append(session.timed(candidate_fn))
    if defer_cooling:
        session.defer_settle()
    else:
        session.settle()
    result = comparison_from_samples(
        paired_means([t * 1e3 for t in base]),
        paired_means([t * 1e3 for t in cand]),
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
