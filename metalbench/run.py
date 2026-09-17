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
Results accumulate in metalbench/results/<chip>.json and <chip>.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metalbench.bridge import HERE, SETS, problems, write_job  # noqa: E402

RESULTS = HERE / "results"
THRESHOLDS = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0)


def chip_name() -> str:
    out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    return out.strip().replace(" ", "-").lower() or "unknown"


def score(report: dict, workload: str) -> dict:
    """One problem's row from its report: speedups against both baselines."""
    step = report.get("step_ms", {}).get(workload, {})
    clocks = report.get("baseline", {}).get("clocks_ms", {}).get(workload, {})
    choice = report.get("baseline", {}).get("choice")
    confirmed = bool(step.get("win_confirmed")) and report.get("session", {}).get("status") != "failed"
    speedup = float(step.get("speedup", 1.0)) if confirmed else 1.0
    plain, compiled = clocks.get("plain"), clocks.get("compiled")
    if choice == "compiled" and plain and compiled:
        vs_compiled, vs_eager = speedup, speedup * plain / compiled
    else:
        vs_compiled, vs_eager = None, speedup
    return {"confirmed": confirmed, "vs_compiled": vs_compiled, "vs_eager": vs_eager,
            "plain_ms": plain, "compiled_ms": compiled,
            "shipped_regions": sum(1 for r in report.get("regions", []) if r.get("s")),
            "status": report.get("session", {}).get("status")}


def fast_p(rows: list[dict], key: str) -> dict[str, float]:
    """fast_p = fraction of problems faster than p against that baseline."""
    scored = [r[key] for r in rows if r.get(key) is not None]
    return {f"fast_{p:g}": (sum(1 for s in scored if s > p) / len(scored) if scored else 0.0)
            for p in THRESHOLDS}


def render(results: dict) -> str:
    rows = list(results["problems"].values())
    lines = [f"# MetalBench on {results['chip']}", "",
             f"{len(rows)} problems. Scores are confirmed whole-step speedups; 1.0 means nothing shipped.", "",
             "| baseline | " + " | ".join(f"fast_{p:g}" for p in THRESHOLDS) + " |",
             "|---|" + "---|" * len(THRESHOLDS)]
    for key, label in (("vs_compiled", "compiled MLX (headline)"), ("vs_eager", "eager MLX")):
        fp = fast_p(rows, key)
        lines.append(f"| {label} | " + " | ".join(f"{fp[f'fast_{p:g}']:.2f}" for p in THRESHOLDS) + " |")
    lines += ["", "| problem | set | vs compiled | vs eager | shipped regions | status |", "|---|---|---|---|---|---|"]
    for name, r in sorted(results["problems"].items()):
        vc = f"{r['vs_compiled']:.3f}x" if r.get("vs_compiled") else "n/a"
        lines.append(f"| {name} | {r['set']} | {vc} | {r['vs_eager']:.3f}x | {r['shipped_regions']} | {r['status']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=SETS + ("all",), default="all")
    parser.add_argument("--only", help="comma-separated problem names")
    parser.add_argument("--budget-per-region", type=int, default=4)
    parser.add_argument("--budget-total", type=int, default=8)
    parser.add_argument("--judge", default="claude-cli")
    parser.add_argument("--rerun", action="store_true", help="rerun problems already scored")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per job")
    args = parser.parse_args(argv)

    chosen = problems(SETS if args.set == "all" else (args.set,))
    if args.only:
        chosen = {n: chosen[n] for n in args.only.split(",")}
    chip = chip_name()
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{chip}.json"
    results = json.loads(path.read_text()) if path.exists() else {"chip": chip, "problems": {}}
    for name, problem in chosen.items():
        if name in results["problems"] and not args.rerun:
            continue
        manifest = write_job(problem, args.budget_per_region, args.budget_total)
        work = HERE / f"work-{problem.set}-{name}"
        if work.exists():
            subprocess.run(["rm", "-rf", str(work)], check=True)
        print(f"[metalbench] {problem.set}/{name}", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", "autotuner.cli", "run", str(manifest),
                               "--judge", args.judge, "--work-dir", str(work)],
                              capture_output=True, text=True, timeout=args.timeout,
                              cwd=Path(__file__).resolve().parents[1])
        report_path = work / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        row = score(report, name)
        row.update(set=problem.set, minutes=round((time.time() - t0) / 60, 1), exit_code=proc.returncode)
        if proc.returncode != 0:
            row["error"] = (proc.stderr.strip().splitlines() or ["?"])[-1][:300]
        results["problems"][name] = row
        path.write_text(json.dumps(results, indent=2) + "\n")
        (RESULTS / f"{chip}.md").write_text(render(results))
        print(f"  vs compiled {row['vs_compiled']}  vs eager {row['vs_eager']:.3f}  ({row['minutes']} min, exit {proc.returncode})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
