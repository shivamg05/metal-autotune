"""Shared gates for measurement-sensitive tests.

Two reasons this machine may not be able to measure: other processes
loading the CPU (the load average shows those; a process holding the GPU
does not show there at all), and a slow GPU, whether thermally throttled
(the fanless chip drops about 4x after sustained work and recovers with idle
cooling) or held by another process. Each gate skips by default so the suite
stays honest about what it verified; set AUTOTUNE_REQUIRE_MEASUREMENT=1 to
make a gated test fail instead, for a run that must prove the timing
claims."""

import os
import time

import mlx.core as mx
import pytest

from autotuner.trace import Tracer

_BANDWIDTH_FLOOR_GBPS = 30.0  # same floor as measure.peaks plausibility
_probe_cache: dict = {}
_current_tracer: Tracer | None = None


def current_tracer() -> Tracer:
    """The Tracer the running test module installed through tracer_for_module."""
    assert _current_tracer is not None, "this module needs `_module_tracer = tracer_for_module()`"
    return _current_tracer


def tracer_for_module():
    """One installed Tracer per test module: up before its first test, restored
    and checked after its last. Any single test can then run alone, and no
    module leaves the patch surface wrapped for the next one."""
    @pytest.fixture(scope="module", autouse=True)
    def _module_tracer():
        global _current_tracer
        tracer = Tracer()
        tracer.install()
        _current_tracer = tracer
        yield tracer
        tracer.uninstall()
        _current_tracer = None
        assert tracer.verify_restored() == []
    return _module_tracer


def _cannot_measure(reason: str) -> None:
    if os.environ.get("AUTOTUNE_REQUIRE_MEASUREMENT"):
        pytest.fail(f"this run must measure, but {reason}")
    pytest.skip(reason)


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
        _cannot_measure(f"the GPU reads {gbps:.0f} GB/s (floor {_BANDWIDTH_FLOOR_GBPS:.0f}): "
                        "throttled or held by another process")


def require_quiet_load(threshold: float = 3.0):
    load = os.getloadavg()[0]
    if load > threshold:
        _cannot_measure(f"the machine is too loaded to measure (load average {load:.1f})")


def arrays_by_path(root_obj) -> dict[str, mx.array]:
    """path -> array for every array reachable from a model: the inverse of
    the snapshot walk, so a test can bind a trace's weights for replay."""
    from autotuner.trace.walk import snapshot_arrays

    by_id = snapshot_arrays(root_obj)
    out: dict[str, mx.array] = {}
    seen: set[int] = set()

    def visit(obj):
        if id(obj) in seen or getattr(obj, "_trace_internal", False):
            return
        seen.add(id(obj))
        if isinstance(obj, mx.array):
            if id(obj) in by_id:
                out.setdefault(by_id[id(obj)], obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                visit(v)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                visit(v)
        elif callable(obj):
            for cell in getattr(obj, "__closure__", None) or ():
                visit(cell.cell_contents)
        if hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                visit(v)

    visit(root_obj)
    return out
