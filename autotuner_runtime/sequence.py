"""Consecutive model calls, shared by the harness and exported bundles."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict
from typing import Callable

import mlx.core as mx
from mlx.utils import tree_flatten

from .stats import comparison_from_samples


def make_sequence(step, inputs, steps: int):
    """A run of consecutive model calls on the same inputs, as one graph. Each
    call's smallest input carries a zero taken from the previous call's first
    output, so no call can start before the last one ends and the GPU runs
    the whole run back to back; calls evaluated one at a time leave it idle
    between them, and a step of a millisecond never reaches full clock that
    way. Every call's outputs are returned, so a discarded output cannot
    turn into missing GPU work."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("sequence steps must be a positive integer")
    carriers = [i for i, a in enumerate(inputs) if isinstance(a, mx.array) and a.dtype != mx.bool_]
    link_at = min(carriers, key=lambda i: inputs[i].nbytes) if carriers else None

    def run():
        outs, link = [], None
        for _ in range(steps):
            args = list(inputs)
            if link is not None:
                args[link_at] = inputs[link_at] + link
            out = step(*args)
            outs.append(out)
            first = next((a for _, a in tree_flatten(out) if isinstance(a, mx.array)), None)
            if link_at is not None and first is not None:
                link = (first.reshape(-1)[0] * 0).astype(inputs[link_at].dtype)
        return outs

    return run


# A run shorter than this heats nothing, the same chunk size the harness session paces by.
PACING_CHUNK_S = 0.25


def compare_sequences(session, baseline, candidate, warm_baseline, warm_candidate,
                      *, pairs=4, warmup_steps=3, label="sequence"):
    """Balance AB / BA order, and cool between complete sequences that are long
    enough to heat the chip. A short run is not cooled after: the idle would
    put the next run on ramped-down clocks the warm-up does not fully recover
    (a 6.3 ms run read 8.9 ms, and a 1.4x faster kernel read slower).

    Calibrate the clock ramp once per arm, then use the larger warmup count
    for *both* arms before each sequence. This avoids a fixed short warmup
    timing tiny workloads on a GPU that has not ramped up yet.
    """
    if pairs < 4 or pairs % 2 or warmup_steps < 1:
        raise ValueError("sequences need an even number of pairs >= 4 and positive warmup steps")
    session.settle()
    warm_count = warmup_steps
    for warm in (warm_baseline, warm_candidate):
        work_s = None
        try:
            times = session.warm_until_stable(warm)
            warm_count = max(warm_count, len(times) + 1)
            work_s = sum(times) * getattr(warm, "steps", 1)
        finally:
            if work_s is None or work_s >= PACING_CHUNK_S:  # a short calibration is not cooled either
                session.settle()
    base, cand, observations = [], [], []
    for pair in range(pairs):
        order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        for arm in order:
            warm, run = ((warm_baseline, baseline) if arm == "baseline"
                         else (warm_candidate, candidate))
            session.log("sequence_start", workload=label, pair=pair, arm=arm,
                        warmup_steps=warm_count)
            elapsed_ms = None
            try:
                for _ in range(warm_count):
                    session.timed(warm)
                elapsed_ms = session.timed(run) * 1000
                (base if arm == "baseline" else cand).append(elapsed_ms)
                observations.append({"pair": pair, "arm": arm, "elapsed_ms": elapsed_ms})
                session.log("sequence_done", workload=label, **observations[-1])
            finally:
                if elapsed_ms is None or elapsed_ms >= PACING_CHUNK_S * 1000:
                    session.settle()
    session.settle()
    comparison = comparison_from_samples(base, cand)
    return {
        "protocol": "whole_sequences_balanced_ab_ba",
        "warmup_steps": warm_count,
        "pairs": pairs,
        "observations": observations,
        "timing": asdict(comparison),
        "baseline_sequence_ms": comparison.median_baseline_ms,
        "candidate_sequence_ms": statistics.median(cand),
        "speedup": 1 / comparison.median_ratio,
        "latency_reduction_pct": 100 * (1 - comparison.median_ratio),
        "win_confirmed": comparison.wins_by(0.0),
        "regression_confirmed": comparison.loses_by(0.0),
    }


class BenchSession:
    """A timing session for a standalone bundle, with no harness present.

    The harness's autotuner/measure/session.py Session is the authority; this
    mirrors its rules: one timing recipe (synchronize, run, evaluate the
    outputs, synchronize), timed GPU work accrues debt that settle() pays as a
    3x idle, and warming throws the first call away then samples until the
    reading stops falling (an improvement beats the best so far by more than
    max(1% of best, 100 us); stop after 10 samples without one, cap 60).
    """

    DUTY_IDLE_FACTOR = 3.0
    WARM_STABLE_RTOL = 0.01
    WARM_STABLE_ABS_S = 100e-6
    WARM_CAP = 60
    WARM_PATIENCE = 10

    def __init__(self, clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep):
        self._clock = clock
        self._sleep = sleep
        self._debt_s = 0.0

    def timed(self, fn: Callable[[], object]) -> float:
        mx.synchronize()
        t0 = self._clock()
        outs = fn()
        mx.eval(outs)
        mx.synchronize()
        elapsed = self._clock() - t0
        self._debt_s += elapsed
        return elapsed

    def settle(self) -> None:
        if self._debt_s <= 0:
            return
        idle = self.DUTY_IDLE_FACTOR * self._debt_s
        self._debt_s = 0.0
        self._sleep(idle)

    def warm_until_stable(self, fn: Callable[[], object]) -> list[float]:
        self.timed(fn)
        times: list[float] = []
        best = None
        stale = 0
        for _ in range(self.WARM_CAP):
            t = self.timed(fn)
            times.append(t)
            improved = best is None or best - t > max(self.WARM_STABLE_RTOL * best, self.WARM_STABLE_ABS_S)
            stale = 0 if improved else stale + 1
            best = t if best is None else min(best, t)
            if stale >= self.WARM_PATIENCE:
                break
        return times

    def log(self, kind: str, **row: object) -> None:
        pass
