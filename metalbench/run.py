"""Run MetalBench problems through autotune and score them KernelBench-style.

Each problem is one job. Its score is the whole-step speedup the job confirmed
end to end, 1.0 when nothing shipped or the job failed, reported against two
baselines: the compiled model (the honest bar, what one mx.compile call gives
for free) and the eager model (what the other benchmarks quote). fast_p is the
fraction of problems faster than p, strictly, so an unchanged model never
counts as a win.

Usage:
  uv run python metalbench/run.py --set common --judge claude-cli
  uv run python metalbench/run.py --only abs,rms_norm_linear --budget-per-region 4 --budget-total 8
Results accumulate in runs/metalbench/results/<chip>.json and <chip>.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metalbench.bridge import HERE, SETS, problems, write_job  # noqa: E402

OUTPUT = HERE.parent / "runs" / "metalbench"
THRESHOLDS = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0)


def chip_name() -> str:
    out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    return out.strip().replace(" ", "-").lower() or "unknown"


def searched_regions(report: dict, workload: str) -> list[dict]:
    """Every part of the model the job searched, costliest first: its best correct
    kernel against the MLX ops it would replace, both clocked side by side, and
    whether it was installed. Under 1 means ours was slower, so MLX's kept running.
    The step's speedup comes from the installed parts only, each weighted by its share."""
    out = []
    for region in sorted(report.get("regions", []), key=lambda g: -(g.get("p") or {}).get(workload, 0.0)):
        rows = [h for h in report.get("hypotheses", []) if h.get("region") == region["fingerprint"]
                and h.get("verdict") != "failed" and h.get("region_ms") and h.get("library_ms")]
        if rows:
            best = max(rows, key=lambda h: h["library_ms"] / h["region_ms"])
            out.append({"ops": "+".join(op.split(".")[-1].strip("_") for op in region["ops"]),
                        "share": (region.get("p") or {}).get(workload, 0.0),
                        "speedup": best["library_ms"] / best["region_ms"], "installed": bool(region.get("s"))})
    return out


def score(report: dict, workload: str) -> dict:
    """One problem's row from its report: the finished model against both
    baselines as measured, and how the best kernel did against the op it replaces.
    A model with nothing installed is the baseline itself, so it reads 1.0."""
    step = report.get("step_ms", {}).get(workload, {})
    clocks = report.get("baseline", {}).get("clocks_ms", {}).get(workload, {})
    choice = report.get("baseline", {}).get("choice")
    failed = report.get("session", {}).get("status") == "failed"
    shipped = sum(1 for r in report.get("regions", []) if r.get("s"))
    speedup = float(step["speedup"]) if shipped and step.get("speedup") and not failed else (None if failed else 1.0)
    if choice == "compiled":
        vs_compiled = speedup
        vs_eager = None if failed or step.get("speedup_vs_plain") is None else float(step["speedup_vs_plain"])
    else:  # shipped against eager: the finished model runs eager plus its kernels
        vs_eager = speedup
        vs_compiled = None if failed or step.get("speedup_vs_compiled") is None else float(step["speedup_vs_compiled"])
    return {"vs_compiled": vs_compiled, "vs_eager": vs_eager, "confirmed": bool(step.get("win_confirmed")) and not failed,
            "regions": searched_regions(report, workload),
            "plain_ms": clocks.get("plain"), "compiled_ms": clocks.get("compiled"),
            "shipped_regions": shipped, "status": report.get("session", {}).get("status")}


def fast_p(rows: list[dict], key: str) -> dict[str, float]:
    """fast_p = fraction of all problems faster than p against that baseline; a job
    that failed has no score and counts as not faster, as KernelBench counts it."""
    scored = [r.get(key) or 0.0 for r in rows]
    return {f"fast_{p:g}": (sum(1 for s in scored if s > p) / len(scored) if scored else 0.0)
            for p in THRESHOLDS}


def render(results: dict) -> str:
    rows = list(results["problems"].values())
    lines = [f"# MetalBench on {results['chip']}", "",
             f"{len(rows)} problems. Scores are the finished model's whole-step speedup as measured, paired "
             f"against each baseline; a model with nothing installed is the baseline, 1.0. Kernels install only "
             f"when they beat the selected baseline. Both columns compare the finished model with the "
             f"named execution mode. The kernel column lists every part of the model the job searched: its "
             f"best kernel against the MLX ops it replaces, its share of the step, and whether it was installed. "
             f"Only installed parts move the step.", "",
             "| baseline | " + " | ".join(f"fast_{p:g}" for p in THRESHOLDS) + " |",
             "|---|" + "---|" * len(THRESHOLDS)]
    for key, label in (("vs_compiled", "compiled MLX (headline)"), ("vs_eager", "eager MLX")):
        fp = fast_p(rows, key)
        lines.append(f"| {label} | " + " | ".join(f"{fp[f'fast_{p:g}']:.2f}" for p in THRESHOLDS) + " |")
    lines += ["", "| problem | set | vs compiled | vs eager | our best kernel per part vs the MLX ops | shipped regions | status |",
              "|---|---|---|---|---|---|---|"]
    ratio = lambda v: f"{v:.3f}x" if v else "n/a"
    for name, r in sorted(results["problems"].items()):
        kernel = "; ".join(f"{k['ops']} {k['speedup']:.2f}x ({k['share']:.0%} of the step"
                           f"{', installed' if k['installed'] else ''})" for k in r.get("regions") or []) or "n/a"
        lines.append(f"| {name} | {r['set']} | {ratio(r.get('vs_compiled'))} | {ratio(r.get('vs_eager'))} | {kernel} "
                     f"| {r['shipped_regions']} | {r['status']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=SETS + ("all",), default="all")
    parser.add_argument("--only", help="comma-separated problem names")
    parser.add_argument("--budget-per-region", type=int, default=4)
    parser.add_argument("--budget-total", type=int, default=8)
    parser.add_argument("--baseline", choices=("plain", "compiled"), default="plain",
                        help="what a kernel must beat to be installed: plain eager MLX, the bar the other "
                             "kernel benchmarks use (default), or the model under mx.compile, the harder bar")
    parser.add_argument("--judge", default="claude-cli")
    parser.add_argument("--judge-effort", default="low",
                        help="thinking effort for the claude-cli judge; pinned so a CLI update cannot change it")
    parser.add_argument("--judge-model", default=None, help="judge model id; defaults to the provider's default")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT,
                        help="directory for generated jobs, logs and scoreboards (default: runs/metalbench)")
    parser.add_argument("--tag", default=None,
                        help="suffix for the results file, so runs with different judges keep separate scoreboards")
    parser.add_argument("--rerun", action="store_true", help="rerun problems already scored")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per job")
    args = parser.parse_args(argv)

    chosen = problems(SETS if args.set == "all" else (args.set,))
    if args.only:
        chosen = {n: chosen[n] for n in args.only.split(",")}
    chip = chip_name()
    output = args.output_dir.resolve()
    results_dir = output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    session = output / "jobs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    stem = ".".join(filter(None, [chip, "eager" if args.baseline == "plain" else None, args.tag]))
    path = results_dir / f"{stem}.json"
    results = json.loads(path.read_text()) if path.exists() else {"chip": chip, "problems": {}}
    for name, problem in chosen.items():
        if name in results["problems"] and not args.rerun:
            continue
        manifest = write_job(problem, args.budget_per_region, args.budget_total,
                             out=session / "generated", baseline=args.baseline)
        work = session / f"{problem.set}-{name}"
        print(f"[metalbench] {problem.set}/{name}", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", "autotuner.cli", "run", str(manifest),
                               "--judge", args.judge, "--judge-effort", args.judge_effort,
                               *(["--model", args.judge_model] if args.judge_model else []),
                               "--work-dir", str(work)],
                              capture_output=True, text=True, timeout=args.timeout,
                              cwd=Path(__file__).resolve().parents[1])
        work.mkdir(parents=True, exist_ok=True)
        (work / "console.log").write_text(proc.stdout + "\n" + proc.stderr)
        report_path = work / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if proc.returncode != 0:
            report.setdefault("session", {})["status"] = "failed"
        row = score(report, name)
        row.update(work_dir=str(work), set=problem.set, minutes=round((time.time() - t0) / 60, 1), exit_code=proc.returncode)
        if proc.returncode != 0:
            row["error"] = (proc.stderr.strip().splitlines() or ["?"])[-1][:300]
        results["problems"][name] = row
        path.write_text(json.dumps(results, indent=2) + "\n")
        (results_dir / f"{stem}.md").write_text(render(results))
        eager = f"{row['vs_eager']:.3f}" if row["vs_eager"] is not None else "n/a"
        print(f"  vs compiled {row['vs_compiled']}  vs eager {eager}  ({row['minutes']} min, exit {proc.returncode})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
