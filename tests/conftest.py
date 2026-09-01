"""Shared gates for measurement-sensitive tests.

Two independent reasons this machine cannot measure: other processes loading
it (visible in the load average), and thermal throttling on the fanless chip
(invisible to load, ~4x down after sustained GPU work, recovers with idle
cooling). Each gate skips rather than fails, so the suite stays honest about
what it verified."""

import os
import time

import mlx.core as mx
import pytest

_BANDWIDTH_FLOOR_GBPS = 30.0  # same floor as measure.peaks plausibility
_probe_cache: dict = {}


def require_healthy_gpu():
    """Skip when a quick bandwidth probe reads below any plausible figure,
    which on this hardware means thermal throttle. Cached briefly so a test
    module does not pay the probe per test."""
    now = time.monotonic()
    if _probe_cache.get("t", -1e9) + 30 > now:
        gbps = _probe_cache["gbps"]
    else:
        x = mx.random.normal((16 * 1024 * 1024,))
        mx.eval(x)
        mx.eval(x + 1.0)  # warm
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(4):
            mx.eval(x + 1.0)
        mx.synchronize()
        gbps = 4 * (2 * 4 * x.size) / (time.perf_counter() - t0) / 1e9
        _probe_cache.update(t=now, gbps=gbps)
    if gbps < _BANDWIDTH_FLOOR_GBPS:
        pytest.skip(f"GPU reads {gbps:.0f} GB/s, thermally throttled; cannot measure")


def require_quiet_load(threshold: float = 3.0):
    load = os.getloadavg()[0]
    if load > threshold:
        pytest.skip(f"machine too loaded to measure (load average {load:.1f})")
