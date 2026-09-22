"""Duty-cycle resting: what it costs the clock, and what dropping it costs the GPU.

The session rests the GPU for 3x the work it just did, then ramps back before
timing. tools/ramp_after_idle.py showed a 0.75s rest is not recoverable, so
the rest biases the step clock. But the rest exists for a reason: this machine
fell from 92 to 22 GB/s after ~8 minutes of sustained test load.

Two modes, both logging GPU bandwidth as they go so the two costs can be read
against each other:

  interleave  alternates a resting and a non-resting arm inside ONE process,
              each bracketed by a back-to-back reference. Interleaving is the
              point: the earlier cross-process attempt was confounded by the
              reference drifting 38% as the chip heated.
  soak        runs one arm only, to watch how far that regime throttles.

    uv run python tools/rest_vs_norest.py --work-dir /tmp/rvn --cycles 10
    uv run python tools/rest_vs_norest.py --work-dir /tmp/rvn2 --mode soak --arm norest
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

from autotuner.loop import JobRunner
from autotuner.measure.clocks import step_clock
from autotuner.measure.peaks import measure_bandwidth
from autotuner.measure.session import Session


def _no_judge(region):
    raise AssertionError("no region should open")


def block(fn, n: int) -> float:
    """Wall ms per step, n steps back to back: the state a generation loop sees."""
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t0) * 1000 / n


def thermal_state() -> int:
    """macOS thermal pressure, 0 nominal to 3 critical, readable without root."""
    import subprocess
    out = subprocess.run(["osascript", "-l", "JavaScript", "-e",
                          "ObjC.import('Foundation'); $.NSProcessInfo.processInfo.thermalState"],
                         capture_output=True, text=True, timeout=10).stdout.strip()
    return int(out) if out.isdigit() else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.yaml")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--mode", choices=("interleave", "soak"), default="interleave")
    ap.add_argument("--arm", choices=("rest", "norest"), default="norest", help="soak only")
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--block", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runner = JobRunner(args.manifest, args.work_dir, judge_factory=_no_judge)
    runner.load_model()
    runner.trace_workloads()
    tensors = runner.tensors[runner.manifest.workloads[0].name]
    runner.tracer.uninstall()
    mx.clear_cache()
    step = lambda: runner.baseline_model(*tensors)

    # sessions live across cycles so pacing debt behaves as it does in a real job
    arms = {"rest": Session(duty_idle_factor=3.0), "norest": Session(duty_idle_factor=0.0)}
    probe = Session(duty_idle_factor=0.0)   # probing must not add rests of its own
    out = Path(args.out) if args.out else Path(args.work_dir) / "rest_vs_norest.jsonl"
    rows = []

    for _ in range(3):                      # reach the running state before cycle 1
        block(step, args.block)

    order = ("rest", "norest")
    for cycle in range(args.cycles):
        names = [args.arm] if args.mode == "soak" else list(
            order if cycle % 2 == 0 else order[::-1])
        bw = measure_bandwidth(probe, samples=3)
        ref_before = block(step, args.block)
        for name in names:
            session = arms[name]
            idled_before = session.idled_s
            t0 = time.perf_counter()
            clock = step_clock(session, step, reps=args.reps)
            wall = time.perf_counter() - t0
            ref_after = block(step, args.block)
            ref = (ref_before + ref_after) / 2
            row = {
                "cycle": cycle, "arm": name, "bandwidth_gbps": round(bw, 1), "thermal": thermal_state(),
                "tool_ms": round(clock.median_ms, 3), "ref_ms": round(ref, 3),
                "ratio": round(clock.median_ms / ref, 4),
                "idled_s": round(session.idled_s - idled_before, 3),
                "wall_s": round(wall, 2),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            ref_before = ref_after

    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\nwrote {out}")
    for name in sorted({r["arm"] for r in rows}):
        a = [r for r in rows if r["arm"] == name]
        ratios = [r["ratio"] for r in a]
        tools = [r["tool_ms"] for r in a]
        print(f"\n{name}: n={len(a)}")
        print(f"  bias   tool/reference  median {st.median(ratios):.3f}"
              f"  range {min(ratios):.3f}-{max(ratios):.3f}")
        print(f"  spread of the tool clock  {(max(tools)/min(tools) - 1) * 100:.1f}%"
              f"   ({min(tools):.2f}-{max(tools):.2f} ms)")
        print(f"  idle paid {sum(r['idled_s'] for r in a):.1f} s"
              f" for {sum(r['wall_s'] for r in a):.1f} s of measuring")
    bws = [r["bandwidth_gbps"] for r in rows]
    print(f"\nGPU bandwidth over the run: {bws[0]:.0f} -> {bws[-1]:.0f} GB/s"
          f"  (min {min(bws):.0f}, max {max(bws):.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
