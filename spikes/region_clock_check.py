"""Is the region clock reproducible across processes, and by how much did the
warm/ramp fix change it?

The step clock is validated (5.713-5.821 ms over four processes, 1.9% apart,
against a true 5.6-5.8). The region clock is the one the spec makes the ship
decision on, so it has to be at least as trustworthy. It shares Session's
warm/ramp, so the fix should reach it, but nothing has measured that.

Prints one JSON line: the priced per-copy cost of the top regions plus the step
clock. Run it several times, with and without the fix, and compare the spread.

    uv run python spikes/region_clock_check.py --work-dir /tmp/rcc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotuner.loop import JobRunner


def _no_judge(region):
    raise AssertionError("no region should open")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.yaml")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    runner = JobRunner(args.manifest, args.work_dir, judge_factory=_no_judge)
    runner.load_model()
    runner.trace_workloads()
    ranked = runner.capture_and_price(runner.build_regions())

    workload = runner.manifest.workloads[0].name
    print(json.dumps({
        "step_ms": round(runner.step_ms[workload], 4),
        "regions": {r.fingerprint[:8]: round(r.t_rep_ms.get(workload, 0.0), 5)
                    for r in ranked[:args.top]},
        "shares": {r.fingerprint[:8]: round(r.p.get(workload, 0.0), 4)
                   for r in ranked[:args.top]},
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
