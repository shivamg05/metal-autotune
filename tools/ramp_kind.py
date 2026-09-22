"""Does the ramp fail because of the idle, or because of the kind of work it uses?

ramp_after_idle.py concluded a 0.75s idle is "unrecoverable": 565 ms of work
left the step at 7.6 ms against a steady 5.6. But that ramp was built from
`time_once` samples, which synchronize before and after every step, so the GPU
sits idle between samples while Python builds the next graph. A GPU picks its
clocks from how busy it is kept. That ramp may itself be what holds the clocks
down, in which case the tool can never warm up, because Session.timed warms
with the very same stop-start pattern it later measures with.

Same idle, two ramp styles, equal GPU work, order alternated:

  sync       repeated time_once(step): synchronize, run, eval, synchronize
  sustained  mx.eval(step()) back to back, nothing between: what inference does

If sustained recovers and sync does not, the bug is the ramp's shape, not the
idle, and the fix is one line in Session.

    uv run python tools/ramp_kind.py --work-dir /tmp/rk
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from autotuner.loop import JobRunner
from autotuner.measure.session import time_once


def _no_judge(region):
    raise AssertionError("no region should open")


def ramp_sync(step, work_ms: float) -> float:
    """The tool's warm-up: isolated synchronized samples, GPU idle between."""
    done = 0.0
    while done < work_ms:
        done += time_once(step) * 1e3
    return done


def ramp_sustained(step, work_ms: float, chunk: int = 20) -> float:
    """Back to back, no synchronize between steps: what a generation loop does."""
    done = 0.0
    while done < work_ms:
        t0 = time.perf_counter()
        for _ in range(chunk):
            mx.eval(step())
        done += (time.perf_counter() - t0) * 1e3
    return done


def reading(step, n: int = 5) -> float:
    """What the tool would report right now: median of n timed samples."""
    return st.median(time_once(step) * 1e3 for _ in range(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.yaml")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--idle", type=float, default=0.75)
    ap.add_argument("--ramp-ms", type=float, default=300.0)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    runner = JobRunner(args.manifest, args.work_dir, judge_factory=_no_judge)
    runner.load_model()
    runner.trace_workloads()
    tensors = runner.tensors[runner.manifest.workloads[0].name]
    runner.tracer.uninstall()
    mx.clear_cache()
    step = lambda: runner.baseline_model(*tensors)

    ramp_sustained(step, 1500)
    steady = reading(step, 9)
    print(f"steady (after sustained work): {steady:.3f} ms\n")

    styles = {"sync": ramp_sync, "sustained": ramp_sustained}
    got: dict[str, list[float]] = {k: [] for k in styles}
    for trial in range(args.trials):
        names = list(styles) if trial % 2 == 0 else list(styles)[::-1]
        for name in names:
            time.sleep(args.idle)
            after_idle = reading(step, 3)
            styles[name](step, args.ramp_ms)
            r = reading(step)
            got[name].append(r)
            print(f"trial {trial}  idle {args.idle}s -> {after_idle:6.2f} ms, "
                  f"then {args.ramp_ms:.0f} ms of {name:<9} ramp -> {r:6.2f} ms")

    print()
    for name, rs in got.items():
        print(f"{name:<10} after ramp: {[round(x, 2) for x in rs]}  "
              f"median {st.median(rs):.3f} ms   {st.median(rs) / steady:.2f}x steady")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
