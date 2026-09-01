"""A machine whose GPU speed changes under the measurement.

Every timing test in this suite runs on a real GPU that happens to be healthy.
The failure this fixture reproduces is the other case, measured live on the
reference machine: bandwidth flipping between 15.7 and 27.8 GB/s on an
eight-second cycle, and sitting 3.2x under its own morning figure. Under that,
any number obtained by dividing two separately timed quantities is fiction.

DriftingSession replaces the clock, not the GPU: functions still run, but the
time reported for one is its declared healthy cost scaled by whatever the
machine is doing at that instant. That makes the failure deterministic, fast,
and free of a GPU, so a test can assert what a measurement reports on a machine
nobody could otherwise hold still.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx

from autotuner.measure.session import Session, time_once

_TOKEN = mx.array([0.0])


def costed(seconds: float) -> Callable[[], object]:
    """A function that 'costs' this many seconds on a healthy machine."""

    def fn():
        return _TOKEN

    fn.healthy_cost_s = seconds
    return fn


def steady(factor: float):
    """A machine running at a constant fraction of its healthy speed."""
    return lambda _call: factor


def ramp(start: float = 1.0, end: float = 4.0, over: int = 60):
    """Thermal drift: the machine slows smoothly and stays slow."""
    def factor(call: int) -> float:
        return start + (end - start) * min(call, over) / over
    return factor


def flip(low: float = 1.0, high: float = 2.5, every: int = 3):
    """Power-state drift: the machine alternates between two discrete speeds.
    This is what the reference machine actually does, and it is the harder
    case, because it moves within a measurement block as well as between."""
    def factor(call: int) -> float:
        return high if (call // every) % 2 else low
    return factor


class DriftingSession(Session):
    """A Session whose clock reports the drifting machine's times. Pacing idles
    are skipped: they exist to let a real chip cool, and here the drift model
    decides the machine's speed, not the sleeping."""

    def __init__(self, factor: Callable[[int], float]):
        super().__init__(sleep=lambda _s: None)
        self._factor = factor
        self.calls = 0
        self.reported: list[float] = []

    def timed(self, fn: Callable[[], object]) -> float:
        """A function's declared healthy cost, or its real cost when it does
        real work, scaled by what the machine is doing at this instant."""
        cost = getattr(fn, "healthy_cost_s", None)
        if cost is None:
            cost = time_once(fn)
        t = cost * self._factor(self.calls)
        self.calls += 1
        self.reported.append(t)
        self._debt_s += t
        return t

    def fresh_chunk(self, fns) -> None:
        """No pacing: the drift model decides the machine's speed here, not
        the sleeping, and ramp-warm samples would consume the drift schedule."""
        self._debt_s = 0.0

    def warm_until_stable(self, fn, **kwargs) -> list[float]:
        """Warm for real, but off the drift schedule, so a test's numbers never
        depend on how many samples the stability rule happened to take."""
        if getattr(fn, "healthy_cost_s", None) is None:
            for _ in range(2):
                time_once(fn)
        return []
