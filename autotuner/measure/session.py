"""Measurement session: the timing recipe, duty-cycle pacing, warm-until-stable.

After a pacing idle the GPU runs at ramped-down clocks, so a block takes
unmeasured ramp samples until the reading stops falling before anything is
timed. Timing a sample right after an idle reads 1.5-1.6x slow and would bias
whichever arm ran first. Callers pace once per measurement, not between the
samples of one, so every sample is taken in the state the model runs in.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx

from autotuner.log import json_safe, wall_now
from autotuner.sandbox.watchdog import gpu_window

DUTY_IDLE_FACTOR = 3.0
MAX_CHUNK_WORK_S = 0.25
WARM_STABLE_RTOL = 0.01
WARM_STABLE_ABS_S = 100e-6  # 1% of a ~1ms kernel is under dispatch jitter
WARM_CAP = 60
# Readings that must fail to improve before the ramp is called done. The GPU
# ramps in steps, not smoothly: after a 0.75s idle the decode step sat at
# ~19 ms for five samples before dropping to ~9 (spikes/ramp_after_idle.py).
# A smaller patience stops on a plateau and calls a cold machine warm.
WARM_PATIENCE = 10


def time_once(fn: Callable[[], object]) -> float:
    """The one timing recipe: synchronize, run, eval outputs, synchronize.

    fn returns its outputs (an array or a tree of arrays); evaluating them here
    is what defeats laziness for whatever fn computed.
    """
    with gpu_window():
        mx.synchronize()
        t0 = time.perf_counter()
        outs = fn()
        mx.eval(outs)
        mx.synchronize()
        return time.perf_counter() - t0


def _until_flat(timed, fn, rtol: float, abs_s: float, cap: int, patience: int) -> list[float]:
    """Sample until the reading stops falling, and return what was timed.

    Not "until two readings agree": while the GPU clocks are still ramping,
    consecutive readings agree with each other and both sit far above the
    steady state. That is how a paced step clock read 8-16 ms for a model
    that runs at 5.7.
    """
    times: list[float] = []
    best: float | None = None
    stale = 0
    for _ in range(cap):
        t = timed(fn)
        times.append(t)
        stale = 0 if best is None or best - t > max(rtol * best, abs_s) else stale + 1
        best = t if best is None else min(best, t)
        if stale >= patience:
            break
    return times


class Session:
    """Owns pacing debt and the append-only health log for one measuring process.

    Timed samples accrue GPU-work debt. fresh_chunk() is the one gate every
    timed loop calls before a sample block: once debt passes the chunk size it
    pays the idle (3x work) and then ramps the clocks back up unmeasured.
    settle() pays whatever debt a measurement leaves behind.
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        duty_idle_factor: float = DUTY_IDLE_FACTOR,
        max_chunk_work_s: float = MAX_CHUNK_WORK_S,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ):
        self.duty_idle_factor = duty_idle_factor
        self.max_chunk_work_s = max_chunk_work_s
        self._sleep = sleep
        self._now = now
        self._ready_at = 0.0
        self._debt_s = 0.0
        self._t0 = time.perf_counter()
        self._log_path = Path(log_path) if log_path else None
        self.idled_s = 0.0
        self.work_s = 0.0

    def log(self, kind: str, **row: object) -> None:
        if self._log_path is None:
            return
        row = {"t": round(time.perf_counter() - self._t0, 3), "wall": wall_now(),
               "kind": kind, **row}
        with self._log_path.open("a") as f:
            f.write(json.dumps(json_safe(row), allow_nan=False) + "\n")

    def timed(self, fn: Callable[[], object]) -> float:
        self.wait_ready()
        t = time_once(fn)
        self._debt_s += t
        self.work_s += t
        return t

    def off_clock(self, fn: Callable[[], object], *, defer_cooling: bool = False):
        """Account for GPU work that evaluates its own results, then cool.

        Capture and correctness checks run outside comparison clocks, but
        still heat the chip. Their full elapsed cost is a conservative debt.
        Defer only when subsequent GPU work goes through wait_ready(); Python
        preparation can then use the cooling interval before that boundary.
        """
        self.wait_ready()
        mx.synchronize()
        start = time.perf_counter()
        try:
            return fn()
        finally:
            mx.synchronize()
            elapsed = time.perf_counter() - start
            self._debt_s += elapsed
            self.work_s += elapsed
            self.log("off_clock_work", seconds=round(elapsed, 3))
            if defer_cooling:
                self.defer_settle()
            else:
                self.settle()

    def fresh_chunk(self, fn: Callable[[], object]) -> bool:
        """Call once before a block of timed samples. Pays pacing debt when a
        chunk's worth has accrued, then ramps the clocks back up with
        unmeasured samples until the reading stops falling. Ramping is a
        property of the hardware, so one function brings the clocks up for
        every arm about to be measured."""
        cooled = self.wait_ready()
        if self._debt_s < self.max_chunk_work_s and not cooled:
            return False
        idle = self.duty_idle_factor * self._debt_s
        self._debt_s = 0.0
        if idle > 0:
            self._idle(idle)
        ramp = _until_flat(self.timed, fn, WARM_STABLE_RTOL, WARM_STABLE_ABS_S,
                           WARM_CAP, WARM_PATIENCE)
        # an idle past ~0.25s is not fully recoverable, so a ramp that never
        # flattens means the block below it is timing a cold machine: say so
        self.log("ramp", idle_s=round(idle, 3), n=len(ramp), flat=len(ramp) < WARM_CAP,
                 first_ms=ramp[0] * 1e3, last_ms=ramp[-1] * 1e3)
        return len(ramp) < WARM_CAP

    def settle(self) -> None:
        """Pay any remaining debt at the end of a measured phase."""
        self.wait_ready()
        if self._debt_s <= 0:
            return
        idle = self.duty_idle_factor * self._debt_s
        self._debt_s = 0.0
        self._idle(idle)

    def defer_settle(self) -> float:
        """Let CPU work use the cooldown; the next GPU phase pays the remainder."""
        if self._debt_s > 0:
            seconds = self.duty_idle_factor * self._debt_s
            self._debt_s = 0.0
            if seconds > 0:
                self._ready_at = max(self._now(), self._ready_at) + seconds
                self.log("cooling_scheduled", seconds=round(seconds, 3))
        return self._ready_at

    def adopt_cooling(self, ready_at: float) -> None:
        """Accept a worker's deadline on this machine's monotonic clock.

        The worker has synchronized its GPU work before handing it back.
        Process teardown, parsing and judge thinking consume the same interval;
        transferring a duration instead would restart cooling in the parent.
        """
        if (isinstance(ready_at, bool) or not isinstance(ready_at, (int, float))
                or not math.isfinite(ready_at) or ready_at < 0):
            raise ValueError("worker cooling deadline must be finite and nonnegative")
        self._ready_at = max(self._ready_at, ready_at)
        self.log("cooling_adopted", remaining_s=round(max(0.0, ready_at - self._now()), 3))

    def wait_ready(self) -> bool:
        """Gate GPU work, including subprocess launches. Returns whether cooled."""
        if not self._ready_at:
            return False
        seconds = max(0.0, self._ready_at - self._now())
        if seconds > 0:
            self._idle(seconds)
        # If sleep is interrupted, a caller that resumes still owes the remainder.
        self._ready_at = 0.0
        self.log("cooling_reused", remaining_s=round(seconds, 3))
        return True

    def _idle(self, seconds: float) -> None:
        self.log("cooling", seconds=round(seconds, 3))
        if seconds >= 5.0:
            print(f"cooling: {seconds:.1f}s after GPU work", flush=True)
        started = self._now()
        try:
            self._sleep(seconds)
        except BaseException:
            self.idled_s += min(seconds, max(0.0, self._now() - started))
            raise
        self.idled_s += seconds
        self.log("cooling_done", seconds=round(seconds, 3))

    def warm_until_stable(
        self,
        fn: Callable[[], object],
        rtol: float = WARM_STABLE_RTOL,
        abs_s: float = WARM_STABLE_ABS_S,
        cap: int = WARM_CAP,
    ) -> list[float]:
        """Law 3: first call thrown away (Metal compile), then sample until the
        reading stops falling. The absolute floor exists because 1% of a small
        kernel is under dispatch jitter. Returns the kept warmup timings."""
        first = self.timed(fn)
        times = _until_flat(self.timed, fn, rtol, abs_s, cap, WARM_PATIENCE)
        self.log("warm", n=len(times) + 1, stable=len(times) < cap,
                 first_ms=first * 1e3, work_s=round(first + sum(times), 3), last_ms=times[-1] * 1e3)
        return times
