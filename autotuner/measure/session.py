"""Measurement session: the timing recipe, duty-cycle pacing, warm-until-stable.

Plan section 6 laws 2, 3, and 6 live here as invariants, with two M0 spike
refinements baked in (spike_10): pacing is chunk-granular, and after every
pacing idle the GPU runs at ramped-down clocks, so each new chunk takes
unmeasured ramp-warm samples before anything is timed. Timing a sample right
after an idle reads 1.5-1.6x slow and would bias whichever arm ran first.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import mlx.core as mx

from autotuner.log import wall_now

DUTY_IDLE_FACTOR = 3.0
MAX_CHUNK_WORK_S = 0.25
RAMP_WARM_SAMPLES = 2
WARM_STABLE_RTOL = 0.01
WARM_STABLE_ABS_S = 100e-6  # spike_10: 1% of a ~1ms kernel is under dispatch jitter
WARM_CAP = 30


def time_once(fn: Callable[[], object]) -> float:
    """The one timing recipe (law 6): synchronize, run, eval outputs, synchronize.

    fn returns its outputs (an array or a tree of arrays); evaluating them here
    is what defeats laziness for whatever fn computed.
    """
    mx.synchronize()
    t0 = time.perf_counter()
    outs = fn()
    mx.eval(outs)
    mx.synchronize()
    return time.perf_counter() - t0


class Session:
    """Owns pacing debt and the append-only health log for one measuring process.

    Timed samples accrue GPU-work debt. fresh_chunk() is the one gate every
    timed loop calls before a sample block: once debt passes the chunk size it
    pays the idle (3x work) and then re-warms the clocks with unmeasured calls
    of the very functions about to be measured.
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        duty_idle_factor: float = DUTY_IDLE_FACTOR,
        max_chunk_work_s: float = MAX_CHUNK_WORK_S,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.duty_idle_factor = duty_idle_factor
        self.max_chunk_work_s = max_chunk_work_s
        self._sleep = sleep
        self._debt_s = 0.0
        self._t0 = time.perf_counter()
        self._log_path = Path(log_path) if log_path else None
        self.idled_s = 0.0

    def log(self, kind: str, **row: object) -> None:
        if self._log_path is None:
            return
        row = {"t": round(time.perf_counter() - self._t0, 3), "wall": wall_now(),
               "kind": kind, **row}
        with self._log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def timed(self, fn: Callable[[], object]) -> float:
        t = time_once(fn)
        self._debt_s += t
        return t

    def fresh_chunk(self, fns: Sequence[Callable[[], object]]) -> None:
        """Call before a block of timed samples of fns. Pays pacing debt when a
        chunk's worth has accrued, then ramp-warms with unmeasured samples so
        the block never starts on ramped-down clocks (spike_10 pacing-clock-ramp)."""
        if self._debt_s < self.max_chunk_work_s:
            return
        idle = self.duty_idle_factor * self._debt_s
        self._debt_s = 0.0
        self.idled_s += idle
        self._sleep(idle)
        for i in range(RAMP_WARM_SAMPLES):
            self._debt_s += time_once(fns[i % len(fns)])  # unmeasured, still GPU work

    def settle(self) -> None:
        """Pay any remaining debt at the end of a measured phase."""
        if self._debt_s <= 0:
            return
        idle = self.duty_idle_factor * self._debt_s
        self._debt_s = 0.0
        self.idled_s += idle
        self._sleep(idle)

    def warm_until_stable(
        self,
        fn: Callable[[], object],
        rtol: float = WARM_STABLE_RTOL,
        abs_s: float = WARM_STABLE_ABS_S,
        cap: int = WARM_CAP,
    ) -> list[float]:
        """Law 3: first call thrown away (Metal compile), then repeat until two
        consecutive timings agree within max(rtol, abs floor). The absolute
        floor exists because 1% of a small kernel is under dispatch jitter
        (spike_10 warm-until-stable). Returns the kept warmup timings."""
        self.timed(fn)
        times: list[float] = []
        for _ in range(cap):
            times.append(self.timed(fn))
            if len(times) >= 2 and abs(times[-1] - times[-2]) <= max(rtol * times[-2], abs_s):
                break
        self.log("warm", n=len(times) + 1, stable=len(times) < cap, last_ms=times[-1] * 1e3)
        return times
