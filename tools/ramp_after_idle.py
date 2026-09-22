"""How much GPU work does it take to get the clocks back after a pacing idle?

The session pays an idle of 3x the work just done, then ramps unmeasured
before timing. The ramp used to be capped at 2 samples / 50 ms, which is why
a step that runs at 5.7 ms was clocked at 8-16 ms. This prints the ramp curve
so the floor is measured rather than guessed.

    uv run python tools/ramp_after_idle.py --work-dir /tmp/ramp
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from autotuner.loop import JobRunner
from autotuner.measure.session import time_once


def _no_judge(region):
    raise AssertionError("no region should open")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.yaml")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--samples", type=int, default=60)
    args = ap.parse_args()

    runner = JobRunner(args.manifest, args.work_dir, judge_factory=_no_judge)
    runner.load_model()
    runner.trace_workloads()
    tensors = runner.tensors[runner.manifest.workloads[0].name]
    runner.tracer.uninstall()
    mx.clear_cache()
    step = lambda: runner.baseline_model(*tensors)

    for _ in range(80):          # reach steady state first
        mx.eval(step())
    steady = min(time_once(step) for _ in range(20)) * 1e3
    print(f"steady: {steady:.3f} ms\n")

    for idle_s in (0.25, 0.75, 2.0):
        time.sleep(idle_s)
        work_ms, curve, reached = 0.0, [], None
        for i in range(args.samples):
            t = time_once(step) * 1e3
            work_ms += t
            curve.append(t)
            if reached is None and t <= steady * 1.05:
                reached = (i + 1, work_ms)
        head = " ".join(f"{t:.1f}" for t in curve[:12])
        print(f"after {idle_s:>4.2f}s idle: {head} ...")
        if reached:
            print(f"  within 5% of steady after {reached[0]} samples "
                  f"= {reached[1]:.0f} ms of GPU work")
        else:
            print(f"  never got within 5% in {work_ms:.0f} ms of work "
                  f"(best {min(curve):.2f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
