"""Does the wrapper cost 23% or 4%? Same model, same process, two clocks.

Two sessions disagree. This one measured the empty wrapper at 1.22-1.25x with
the harness's own paired clock, which paces with idle between samples. Another
measured 1.04x "in the steady state" (steps back to back, no sleeps) and says
the 1.23x is an artifact of timing in a cold state where Python runs 2-3x
slower. A third reading exists in this session's own spikes/why_wrapper_costs.py,
which timed back to back and still got 1.35x.

All three cannot be right. This runs both clocks on one model in one process,
so thermal state, weights and machine are shared and only the clock differs.

    uv run python spikes/steady_vs_paced.py --work-dir /tmp/svp
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from autotuner.bind.certify import find_scope_call
from autotuner.bind.emit import emit_wrapper
from autotuner.bind.swap import install as swap_install
from autotuner.loop import JobRunner, _load_class, _resolve, _safe
from autotuner.measure.clocks import compare
from autotuner.measure.session import Session, time_once

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isolate_shipped import hoist_weights  # noqa: E402


def _no_judge(region):
    raise AssertionError("no region should open")


def block(fn, n: int) -> float:
    """Wall ms per step, n steps back to back, nothing between them."""
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t0) * 1000 / n


def warm_until_flat(fn, n: int = 30, limit: int = 40) -> list[float]:
    """Warm until the reading stops falling, not until two agree."""
    seen = [block(fn, n)]
    for _ in range(limit):
        cur = block(fn, n)
        if cur >= seen[-1] * 0.99:      # stopped falling
            seen.append(cur)
            return seen
        seen.append(cur)
    return seen


def steady_pairs(base, pat, pairs: int, n: int) -> list[float]:
    """ABBA at block level, no idle anywhere. Returns per-pair pat/base."""
    out = []
    for _ in range(pairs // 2):
        a1 = block(base, n); b1 = block(pat, n)
        b2 = block(pat, n);  a2 = block(base, n)
        out += [b1 / a1, b2 / a2]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.yaml")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--region", default="ad5dbb4cfe82c939")
    ap.add_argument("--hoisted", action="store_true")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--block", type=int, default=30)
    ap.add_argument("--duty", type=float, default=None,
                    help="override the pacing idle factor; 0 disables resting")
    args = ap.parse_args()

    session = None if args.duty is None else Session(duty_idle_factor=args.duty)
    runner = JobRunner(args.manifest, args.work_dir, judge_factory=_no_judge,
                       session=session)
    runner.load_model()
    runner.trace_workloads()
    regions = {r.fingerprint: r for r in runner.build_regions()}
    region = regions[args.region]
    tensors = runner.tensors[runner.manifest.workloads[0].name]
    runner.tracer.uninstall()
    mx.clear_cache()

    plain = lambda: runner.baseline_model(*tensors)
    ramp = warm_until_flat(plain, args.block)
    print(f"warm ramp (ms/step): {[round(x, 2) for x in ramp]}")
    print(f"steady-state step: {ramp[-1]:.3f} ms\n")

    scopes, seen = [], []
    for m in region.members:
        trace = runner.traces[m.workload]
        scope = find_scope_call(trace, m.scope_stack)
        path = scope.address.rsplit("@", 1)[0]
        if path not in seen:
            seen.append(path)
            scopes.append((path, trace, scope))
    try:
        runner.tracer.patcher.install()
    except RuntimeError:
        pass
    for path, trace, scope in scopes:
        em = emit_wrapper(trace, scope, [], f"Id_{_safe(path)}")
        if args.hoisted:
            em.source = hoist_weights(em.source)
        swap_install(runner.model, path, _load_class(em)(_resolve(runner.model, path), {}))
    runner.tracer.uninstall()
    mx.clear_cache()

    wrapped = lambda: runner.model(*tensors)
    warm_until_flat(wrapped, args.block)

    r = steady_pairs(plain, wrapped, args.pairs, args.block)
    steady = st.median(r)
    print(f"STEADY  (back to back, no idle): ratio {steady:.4f}  "
          f"pairs {[round(x, 3) for x in r]}")

    # the session only paces once a chunk's worth of work has accrued; in a real
    # job that debt is always there, so prime it or the comparison never idles
    while runner.session._debt_s < runner.session.max_chunk_work_s:
        runner.session.timed(plain)
    paced = compare(runner.session, plain, wrapped, pairs=32)
    print(f"PACED   (harness compare, 32 pairs): ratio {paced.median_ratio:.4f}  "
          f"baseline {paced.median_baseline_ms:.3f} ms  sigma {paced.sigma_ms:.3f}")

    print("\n" + json.dumps({
        "hoisted": args.hoisted,
        "duty": args.duty,
        "steady_step_ms": ramp[-1],
        "steady_ratio": steady,
        "paced_ratio": paced.median_ratio,
        "paced_baseline_ms": paced.median_baseline_ms,
        "scopes": len(scopes),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
