"""Where the identity wrapper's ~68 us per layer per token goes.

isolate_shipped.py showed the generated wrapper costs 1.23x with no kernel in
it. This dumps the wrapper the job would install for one layer and profiles a
run with all 28 installed, so the cost has a named source instead of a guess.

    uv run python spikes/why_wrapper_costs.py --work-dir /tmp/whycost
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from autotuner.bind.certify import find_scope_call
from autotuner.bind.emit import emit_wrapper, scope_nodes
from autotuner.bind.swap import install as swap_install
from autotuner.loop import JobRunner, _load_class, _resolve, _safe


def _no_judge(region):
    raise AssertionError("no region should open")


def steps(fn, n: int) -> float:
    """Wall ms per call, laziness defeated the way the harness defeats it."""
    mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) * 1000 / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.yaml")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--region", default="ad5dbb4cfe82c939")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dump", default="wrapper_dump.py")
    args = ap.parse_args()

    runner = JobRunner(args.manifest, args.work_dir, judge_factory=_no_judge)
    runner.load_model()
    runner.trace_workloads()
    regions = {r.fingerprint: r for r in runner.build_regions()}
    region = regions[args.region]
    workload = runner.manifest.workloads[0].name
    tensors = runner.tensors[workload]

    # one scope per layer; emit the wrapper the job would install there
    scopes, seen = [], set()
    for m in region.members:
        trace = runner.traces[m.workload]
        scope = find_scope_call(trace, m.scope_stack)
        path = scope.address.rsplit("@", 1)[0]
        if path not in seen:
            seen.add(path)
            scopes.append((path, trace, scope))

    first_path, first_trace, first_scope = scopes[0]
    emitted = emit_wrapper(first_trace, first_scope, [], f"Id_{_safe(first_path)}")
    Path(args.dump).write_text(emitted.source)
    nodes = scope_nodes(first_trace, first_scope)
    print(f"scopes: {len(scopes)}   ops replayed per scope: {len(nodes)}")
    print(f"wrapper source: {args.dump} ({len(emitted.source.splitlines())} lines)")

    runner.tracer.uninstall()
    mx.clear_cache()

    before = steps(lambda: runner.model(*tensors), args.steps)

    try:
        runner.tracer.patcher.install()
    except RuntimeError:
        pass
    for path, trace, scope in scopes:
        em = emit_wrapper(trace, scope, [], f"Id_{_safe(path)}")
        swap_install(runner.model, path, _load_class(em)(_resolve(runner.model, path), {}))
    runner.tracer.uninstall()
    mx.clear_cache()

    after = steps(lambda: runner.model(*tensors), args.steps)
    added = after - before
    print(f"\nstep ms: plain {before:.3f} -> wrapped {after:.3f}  (+{added:.3f} ms, "
          f"{added/len(scopes)*1000:.1f} us per scope, "
          f"{added/len(scopes)/max(len(nodes),1)*1000:.2f} us per replayed op)")

    prof = cProfile.Profile()
    prof.enable()
    for _ in range(args.steps):
        mx.eval(runner.model(*tensors))
    mx.synchronize()
    prof.disable()
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(22)
    print("\n=== profile of the wrapped model (tottime) ===")
    print("\n".join(s.getvalue().splitlines()[4:34]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
