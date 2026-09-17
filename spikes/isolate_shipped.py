"""Time each shipped kernel alone in the forward pass, against the untouched model.

Diagnostic for run work-2026-09-02-1952-est, where three kernels each measured
faster than the library in the region clock and each passed the whole-model
veto when installed, yet the model with all three read 1.31x slower at the end.

One arm per process, because the thing under suspicion is accumulated state:

    none      nothing installed; the control, must read ~1.00
    identity  the generated wrapper with no kernel in it, so the replay tax
              alone is visible (certify_identity proves this is numerically
              invisible; nothing has ever measured whether it is free)
    kernel    the region's shipped kernel bound the way the job binds it

Every arm re-times the untouched model interleaved with the patched one, so
drift inside a comparison cancels; median_ratio is what to compare across arms.

    uv run python spikes/isolate_shipped.py --arm kernel --regions ad5dbb4cfe82c939
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

import autotuner.loop as loop
from autotuner.bind.certify import find_scope_call
from autotuner.bind.emit import emit_wrapper
from autotuner.bind.swap import install as swap_install
from autotuner.loop import JobRunner, RegionRun, _load_class, _resolve, _safe
from autotuner.measure.clocks import compare
from autotuner.measure.session import Session
from autotuner_runtime.kernels import load_spec


class _AlwaysPasses:
    """Stands in for run_e2e inside _bind_and_promote: this script measures the
    veto itself, so the bind must not be gated on it. Everything else in the
    bind path (identity certification, wrapper emit, literal retrace) still runs."""

    passed = True
    veto_passed = True
    veto = None
    checks: list = []


def _no_judge(region):
    raise AssertionError(f"a region was opened: {region.fingerprint}")


_WEIGHT_PATH = re.compile(r"self\.wrapped((?:\.[A-Za-z_][A-Za-z0-9_]*)+)")


def hoist_weights(source: str) -> str:
    """Resolve every `self.wrapped.a.b.weight` once instead of per token.

    The generated __call__ re-walks the module tree for each weight it needs,
    and in MLX every hop is a Python __getattr__ over a dict, with the
    wrapper's own __getattr__ stacked on top. The weights do not move, so the
    walk is pure overhead. This rewrites them to indexes into a tuple the
    wrapper resolves in its own __init__, held in the instance __dict__ so it
    is found by normal lookup and never enters the parameter tree.
    `self.wrapped(...)` in the shape-guard fallback has no attribute after it
    and is left alone.
    """
    paths: list[str] = []

    def swap(m):
        path = m.group(1)[1:]
        if path not in paths:
            paths.append(path)
        return f"_b[{paths.index(path)}]"

    out = _WEIGHT_PATH.sub(swap, source)
    if not paths:
        return out
    init = (
        "\n    def __init__(self, wrapped, specs=None):\n"
        "        super().__init__(wrapped, specs)\n"
        f"        _paths = {tuple(paths)!r}\n"
        "        _vals = []\n"
        "        for _p in _paths:\n"
        "            _o = wrapped\n"
        "            for _part in _p.split('.'):\n"
        "                _o = getattr(_o, _part)\n"
        "            _vals.append(_o)\n"
        "        object.__setattr__(self, '_bound', tuple(_vals))\n"
    )
    marker = "\n    def __call__(self"
    i = out.index(marker)
    out = out[:i] + init + out[i:]
    j = out.index(marker)
    eol = out.index("\n", j + 1)
    return out[:eol] + "\n        _b = self._bound" + out[eol:]


def scopes_of(runner: JobRunner, region) -> list[tuple[str, object, object]]:
    """Every (scope_path, trace, scope call) the region's copies land in."""
    out, seen = [], set()
    for m in region.members:
        trace = runner.traces[m.workload]
        scope = find_scope_call(trace, m.scope_stack)
        if scope is None:
            raise SystemExit(f"no scope call for {m.scope_stack!r}")
        path = scope.address.rsplit("@", 1)[0]
        if path and path not in seen:
            seen.add(path)
            out.append((path, trace, scope))
    return out


def install_identity(runner: JobRunner, region, hoisted: bool = False) -> int:
    """The wrapper the job would install, carrying no cut at all."""
    try:
        runner.tracer.patcher.install()
    except RuntimeError:
        pass
    n = 0
    for path, trace, scope in scopes_of(runner, region):
        emitted = emit_wrapper(trace, scope, [], f"Id_{_safe(path)}")
        if hoisted:
            emitted.source = hoist_weights(emitted.source)
        original = _resolve(runner.model, path)
        swap_install(runner.model, path, _load_class(emitted)(original, {}))
        runner.emitted[path] = emitted
        n += 1
    runner.tracer.uninstall()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=(
        "none", "identity", "identity_hoisted", "kernel", "kernel_hoisted"))
    ap.add_argument("--regions", default="", help="comma-separated fingerprints")
    ap.add_argument("--source", default="work-2026-09-02-1952-est",
                    help="the finished run whose shipped kernels are replayed")
    ap.add_argument("--manifest", default="manifest.yaml")
    ap.add_argument("--work-dir", required=True, help="fresh scratch dir for this arm")
    ap.add_argument("--pairs", type=int, default=32)
    args = ap.parse_args()

    fingerprints = [f for f in args.regions.split(",") if f]
    # a region can ship more than once; the last one is what stayed installed
    shipped = {}
    for line in (Path(args.source) / "run.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["kind"] == "shipped":
            shipped[row["fingerprint"]] = row["kernel"]

    runner = JobRunner(args.manifest, args.work_dir, judge_factory=_no_judge)
    runner.load_model()
    runner.trace_workloads()
    ranked = runner.capture_and_price(runner.build_regions())
    by_fp = {r.fingerprint: r for r in ranked}

    loop.run_e2e = lambda *a, **k: _AlwaysPasses()

    if args.arm == "kernel_hoisted":
        # the same transform, inside the production bind path: every wrapper it
        # emits, the identity one it certifies with included, gets hoisted
        _emit = loop.emit_wrapper

        def _hoisted(*a, **k):
            em = _emit(*a, **k)
            em.source = hoist_weights(em.source)
            return em

        loop.emit_wrapper = _hoisted

    scopes_installed = 0
    for fp in fingerprints:
        region = by_fp.get(fp)
        if region is None:
            raise SystemExit(f"{fp} is not a region of this trace")
        if args.arm.startswith("identity"):
            scopes_installed += install_identity(
                runner, region, hoisted=args.arm == "identity_hoisted")
        else:
            spec = load_spec(Path(args.source) / "kernels" / f"{shipped[fp]}.metal")
            if not runner._bind_and_promote(RegionRun(region=region), spec, None):
                raise SystemExit(f"bind failed for {fp}; see {args.work_dir}/run.jsonl")

    # a silent rollback would time an unpatched model and read 1.00, which is
    # indistinguishable from an innocent kernel: prove the install landed
    rows = [json.loads(l) for l in (Path(args.work_dir) / "run.jsonl").read_text().splitlines()]
    bad = [r for r in rows if r["kind"] in ("bind_failed", "certification_failed",
                                            "rollback_error")]
    if bad:
        raise SystemExit(f"bind did not hold: {bad}")
    if args.arm.startswith("kernel") and fingerprints:
        # runner.installed and runner.cuts are filled by _bind_and_promote only
        if not runner.installed:
            raise SystemExit("nothing is installed")
        spans = sum(len(c) for c in runner.cuts.values())
        want = sum(len(by_fp[f].members) for f in fingerprints)
        if spans != want:
            raise SystemExit(f"{spans} spans cut, expected {want}")
    elif args.arm.startswith("identity"):
        if scopes_installed == 0:
            raise SystemExit("no identity wrapper was installed")
    elif runner.installed or runner.emitted:
        raise SystemExit("the control arm has something installed")

    if args.arm != "none":
        # the tool's own rule, not a hand-picked tolerance: a kernel may move
        # the output (production saw max_abs 0.18 against an allowance of 2.67)
        # while an identity wrapper may not
        from autotuner.e2e import preserving_check
        first = runner.manifest.workloads[0].name
        chk = preserving_check(
            lambda: runner.baseline_model(*runner.tensors[first]),
            lambda: runner.model(*runner.tensors[first]), first)
        if not chk.passed:
            raise SystemExit(f"outputs not preserved: max_abs {chk.max_abs} "
                             f"vs allowance {chk.allowance}")

    baseline_arm, patched_arm = runner._timed_arms()
    # The wrapper's cost is mostly Python on the CPU, and idling slows the CPU,
    # so this must be timed in the state the model actually runs in. The pricing
    # phase leaves a large pacing debt; pay it, drive the machine back with
    # sustained work, then time on a fresh session that will not idle mid-run.
    runner.session.settle()
    for _ in range(400):
        mx.eval(baseline_arm())
    c = compare(Session(), baseline_arm, patched_arm, pairs=args.pairs)
    print(json.dumps({
        "arm": args.arm,
        "regions": fingerprints,
        "kernels": [shipped[f] for f in fingerprints] if args.arm.startswith("kernel") else [],
        "scopes_installed": scopes_installed or len(runner.installed),
        "pairs": args.pairs,
        "median_ratio": c.median_ratio,          # patched / untouched, <1 is faster
        "median_baseline_ms": c.median_baseline_ms,
        "median_delta_ms": c.median_delta_ms,
        "sigma_ms": c.sigma_ms,
        "stability": c.stability,
        "max_abs": chk.max_abs if args.arm != "none" else None,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
