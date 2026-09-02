"""Chip peaks: bandwidth, per-dtype flops, launch cost. Microbenched, never
read off a spec sheet.

Law 5: the estimator is the running max over samples spread across the window,
because throttling, contention, and cold start only push observations down.
Recalibration may only raise a peak.
"""

from __future__ import annotations

import re
import statistics
import subprocess
from dataclasses import dataclass, field

import mlx.core as mx

from .session import Session

BANDWIDTH_ELEMENTS = 64 * 1024 * 1024  # fp32: two 256MB buffers in flight
MATMUL_N = 2048
LAUNCH_CHAIN = 256
FALLBACK_LAUNCH_US = 4.0


@dataclass(frozen=True)
class Peaks:
    bandwidth_gbps: float
    flops_gflops: dict[str, float] = field(default_factory=dict)
    launch_us: float = FALLBACK_LAUNCH_US


# floors no Apple-Silicon GPU plausibly sits under (same bounds as the pinned
# physics test); readings below them mean a degraded machine, not a slow chip
# (seen live: work-2026-08-31-halted, a 10x-degraded GPU within one boot)
PLAUSIBLE_MIN_GBPS = 30.0
PLAUSIBLE_MIN_FP32_GFLOPS = 500.0


def best_of(a: Peaks, b: Peaks) -> Peaks:
    """Two readings of the same chip: throttling, contention, and cold
    start only push a reading down, so the higher one is the truer peak and
    the lower launch cost the truer floor."""
    flops = {d: max(a.flops_gflops.get(d, 0.0), b.flops_gflops.get(d, 0.0))
             for d in set(a.flops_gflops) | set(b.flops_gflops)}
    return Peaks(bandwidth_gbps=max(a.bandwidth_gbps, b.bandwidth_gbps),
                 flops_gflops=flops, launch_us=min(a.launch_us, b.launch_us))


_UTILIZATION = re.compile(r'"Device Utilization %"=(\d+)')
BUSY_GPU_PERCENT = 50.0


def gpu_utilization(text: str | None = None) -> float | None:
    """How busy macOS says the GPU is right now, in percent, before this
    process has issued any work of its own; None when it cannot be read. A
    background process holding the GPU shows here and nowhere else: it is
    invisible to the load average and to every peak probe's pairing."""
    if text is None:
        try:
            text = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-d", "4"],
                                  capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    m = _UTILIZATION.search(text)
    return float(m.group(1)) if m else None


def implausible(peaks: Peaks) -> str | None:
    """A one-line reason these peaks cannot be a healthy machine, else None."""
    if peaks.bandwidth_gbps < PLAUSIBLE_MIN_GBPS:
        return (f"measured bandwidth {peaks.bandwidth_gbps:.1f} GB/s is below any "
                f"plausible Apple-Silicon figure (floor {PLAUSIBLE_MIN_GBPS:.0f})")
    fp32 = peaks.flops_gflops.get("float32")
    if fp32 is not None and fp32 < PLAUSIBLE_MIN_FP32_GFLOPS:
        return (f"measured fp32 {fp32:.0f} GFLOP/s is below any plausible "
                f"Apple-Silicon figure (floor {PLAUSIBLE_MIN_FP32_GFLOPS:.0f})")
    return None


def measure_bandwidth(session: Session, samples: int = 6) -> float:
    x = mx.random.normal((BANDWIDTH_ELEMENTS,))
    mx.eval(x)
    bytes_moved = 2 * 4 * BANDWIDTH_ELEMENTS  # read x, write y
    fn = lambda: x + 1.0
    session.warm_until_stable(fn)
    peak = 0.0
    for _ in range(samples):
        session.fresh_chunk((fn,))
        t = session.timed(fn)
        peak = max(peak, bytes_moved / t / 1e9)
    session.settle()
    session.log("peak_bandwidth", gbps=round(peak, 1))
    return peak


def measure_flops(session: Session, dtype: mx.Dtype, samples: int = 6, n: int = MATMUL_N) -> float:
    a = mx.random.normal((n, n)).astype(dtype)
    b = mx.random.normal((n, n)).astype(dtype)
    mx.eval(a, b)
    flops = 2 * n * n * n
    fn = lambda: a @ b
    session.warm_until_stable(fn)
    peak = 0.0
    for _ in range(samples):
        session.fresh_chunk((fn,))
        t = session.timed(fn)
        peak = max(peak, flops / t / 1e9)
    session.settle()
    session.log("peak_flops", dtype=str(dtype), gflops=round(peak, 1))
    return peak


def measure_launch_us(session: Session, chain: int = LAUNCH_CHAIN, samples: int = 5) -> float:
    """Per-kernel dispatch cost from a chain of dependent tiny ops. Each op is
    one kernel; the dependency chain stops anything overlapping. Median here,
    not max: launch cost feeds T_launch as a typical cost, not a capability."""
    x = mx.array([1.0])
    mx.eval(x)

    def fn():
        y = x
        for _ in range(chain):
            y = y + 1.0
        return y

    session.warm_until_stable(fn)
    times = []
    for _ in range(samples):
        session.fresh_chunk((fn,))
        times.append(session.timed(fn))
    session.settle()
    per_launch = statistics.median(times) / chain * 1e6
    session.log("peak_launch", us=round(per_launch, 2))
    return per_launch


def measure_peaks(session: Session, dtypes: tuple[mx.Dtype, ...] = (mx.float32, mx.float16, mx.bfloat16)) -> Peaks:
    """Job-start calibration. Flops peaks are per dtype present in the trace;
    the caller passes the dtypes it actually saw."""
    bandwidth = measure_bandwidth(session)
    flops = {str(d).removeprefix("mlx.core."): measure_flops(session, d) for d in dtypes}
    launch = measure_launch_us(session)
    return Peaks(bandwidth_gbps=bandwidth, flops_gflops=flops, launch_us=launch)
