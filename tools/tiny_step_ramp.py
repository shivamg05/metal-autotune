"""Does warming ever bring a ~1 ms model step to its true clock, or only chaining?

A step this small, called once per sample with a CPU gap after it, is too
little work to bring the GPU clock up after an idle; the reading sits on a
plateau that warm-until-flat mistakes for steady state. See docs/architecture.md for the measurement invariants.

    uv run python tools/tiny_step_ramp.py --work-dir /tmp/tsr [--problem rms_norm_linear]
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from autotuner.measure.session import time_once
from metalbench.bridge import problems, write_job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--problem", default="rms_norm_linear")
    args = ap.parse_args()

    p = problems()[args.problem]
    manifest = write_job(p, 1, 1, out=Path(args.work_dir))
    spec = importlib.util.spec_from_file_location("m", manifest.parent / "model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    model = m.build()
    inputs = [mx.random.normal(s) for s in p.input_shapes]
    mx.eval(inputs)
    mx.eval(model(*inputs))
    print("shapes", p.input_shapes)

    def single():
        return model(*inputs)

    def chained(k):
        def run():
            x, outs = inputs[0], []
            for _ in range(k):
                out = model(x, *inputs[1:])
                outs.append(out)
                first = out[0] if isinstance(out, (list, tuple)) else out
                x = inputs[0] + (first.reshape(-1)[0] * 0).astype(x.dtype)
            return outs
        return run

    def bands(ts, edges=(0, 10, 50, 200, 400)):
        return "  ".join(f"[{a}:{b}] {statistics.median(ts[a:b]) * 1e3:.2f}"
                         for a, b in zip(edges, edges[1:]) if ts[a:b])

    def series(n=400):
        return [time_once(single) for _ in range(n)]

    time.sleep(2.0)
    ts = series()
    print("A) 2 s idle, then 400 single calls back to back (ms per call, median by sample band)")
    print("   first 8:", " ".join(f"{t * 1e3:.2f}" for t in ts[:8]))
    print("  ", bands(ts))

    a = mx.random.normal((4096, 4096))
    mx.eval(a)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        mx.eval(a @ a)
    ts = series()
    mn = min(ts[:5])
    i = next((i for i, t in enumerate(ts) if t > 2 * mn), None)
    print("B) 1 s of 4096-square matmuls, then 400 single calls back to back")
    print("   first 8:", " ".join(f"{t * 1e3:.2f}" for t in ts[:8]))
    print("  ", bands(ts))
    print(f"   first call reading >2x the early ones: index {i}, after {sum(ts[:i]) * 1e3:.0f} ms of calls"
          if i is not None else "   never slowed")

    print("C) 2 s idle, 3 throwaway samples, then 9 samples of k chained steps")
    for k in (1, 5, 20, 80):
        time.sleep(2.0)
        fn = chained(k)
        for _ in range(3):
            time_once(fn)
        s = [time_once(fn) for _ in range(9)]
        med = statistics.median(s) * 1e3
        print(f"   k={k:3d}: sample {med:7.2f} ms -> {med / k:.3f} ms/step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
