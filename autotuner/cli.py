"""autotune run manifest.yaml"""

import argparse
from datetime import datetime
from contextlib import contextmanager
import fcntl
import os
import sys
import tempfile
from pathlib import Path


@contextmanager
def _run_lock():
    """Only one CLI job may measure the shared GPU; crashes release the lock."""
    path = Path(tempfile.gettempdir()) / f"metal-autotune-{os.getuid()}.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another autotune CLI job is running; wait for it to finish") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _output_paths(work_dir, artifact):
    work = Path(work_dir).resolve()
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        raise ValueError(f"work directory is not empty: {work}; choose a fresh directory")
    out = Path(artifact).resolve() if artifact else work / "artifact"
    if out == work or out in work.parents:
        raise ValueError("--artifact must not be the work directory or one of its parents")
    for name in ("kernels", "boundaries", "checkpoints", "judge_io", "run.jsonl",
                 "session.jsonl", "report.json", "judge.jsonl", "candidates.log"):
        reserved = work / name
        if out == reserved or reserved in out.parents:
            raise ValueError(f"--artifact overlaps run files: {reserved}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"artifact destination already exists: {out}; choose a fresh destination")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="autotune")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one optimization job")
    run.add_argument("manifest")
    run.add_argument("--work-dir", default=str(Path("runs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")),
                     help="fresh run directory; defaults to runs/<timestamp>")
    run.add_argument("--artifact", default=None, help="fresh output directory; defaults to <work-dir>/artifact")
    run.add_argument("--model", default=None, help="judge model id; defaults to the selected provider's model")
    run.add_argument("--judge", default="api",
                     help="api: Anthropic SDK (needs ANTHROPIC_API_KEY); "
                          "claude-cli, codex, gemini: that agent's CLI, headless, in an empty dir; "
                          "agent: answer <work-dir>/judge_io by hand (see docs/judge-protocol.md)")
    run.add_argument("--judge-effort", default=None,
                     help="thinking effort for the claude-cli judge (low, medium, high, xhigh, max); "
                          "default low, the level the CLI ran at before its 2026-09-17 update")
    run.add_argument("--judge-cmd", default=None,
                     help="run this exact command as the judge, any headless agent CLI; a "
                          "{system}/{prompt} token is filled in, else the prompt goes on stdin. "
                          "Overrides --judge.")
    finish = sub.add_parser("finish", help="stop new search attempts and finish validation/export")
    finish.add_argument("--work-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "finish":
        work = Path(args.work_dir).resolve()
        if not (work / "run.jsonl").is_file():
            parser.error(f"no run.jsonl in {work}")
        (work / "finish-search.request").write_text("operator requested final validation\n")
        print("Finish requested: the running job will finish its current work, then validate and export if confirmed.")
        return 0

    try:
        args.artifact = _output_paths(args.work_dir, args.artifact)
    except ValueError as error:
        parser.error(str(error))
    with _run_lock():
        return _execute(args, parser)


def _execute(args, parser):

    from .loop import JobRunner
    from .judge.agent import CLI_PRESETS, AgentFileJudge, CliJudge

    if args.judge_cmd:
        import shlex
        judge = CliJudge(shlex.split(args.judge_cmd))
        label = args.judge_cmd
    elif args.judge == "api":
        from .judge.client import AnthropicJudge, DEFAULT_MODEL
        judge = AnthropicJudge(model=args.model or DEFAULT_MODEL)
        label = f"api ({args.model or DEFAULT_MODEL})"
    elif args.judge in CLI_PRESETS:
        from .judge.agent import DEFAULT_EFFORT, claude_argv
        preset = CLI_PRESETS[args.judge]
        effort = args.judge_effort or DEFAULT_EFFORT
        judge = CliJudge(preset(args.model, effort) if preset is claude_argv else preset(args.model))
        label = f"{args.judge} ({args.model or 'provider default'}"
        label += f", effort {effort})" if preset is claude_argv else ")"
    elif args.judge == "agent":
        mailbox = Path(args.work_dir) / "judge_io"
        judge = AgentFileJudge(mailbox)
        print(f"agent judge: answer requests in {mailbox}/ (protocol in docs/judge-protocol.md)")
        label = "agent"
    else:
        parser.error(f"--judge must be api, agent, or one of {sorted(CLI_PRESETS)}, "
                     f"or use --judge-cmd; got {args.judge!r}")

    if isinstance(judge, CliJudge):
        print("Checking judge connection and model access before loading the model...", flush=True)
        try:
            judge.check_available()
        except ValueError as error:
            parser.error(str(error))
        print("Judge readiness check passed.", flush=True)
    judge.transcript = Path(args.work_dir) / "judge.jsonl"
    print(f"judge: {label}")
    print(f"run log: {Path(args.work_dir) / 'run.jsonl'} (one JSON line per event; tail it)")
    print(f"candidates: {Path(args.work_dir) / 'candidates.log'} (one line per attempt)")
    runner = JobRunner(
        args.manifest,
        args.work_dir,
        judge_factory=lambda region: judge,
    )
    try:
        report = runner.run()
        artifact = None
        if report.accepted and runner.final_ok:
            artifact = runner.emit_artifact(args.artifact)
        else:
            report.session.update(status="complete", artifact=None,
                                  outcome="unconfirmed" if report.accepted else "no_improvement")
            report.write(Path(args.work_dir) / "report.json")
    except Exception:
        print(f"job failed: see {args.work_dir}/report.json and the error below", file=sys.stderr)
        raise
    _print_result(report, artifact)
    return 0


def _print_result(report, artifact):
    if artifact:
        shipped = sum(bool(r.get("s")) for r in report.regions)
        print(f"job complete: verified artifact with {shipped}/{len(report.regions)} regions optimized")
    else:
        print("job complete: no confirmed improvement; no artifact produced")
        if report.accepted:
            print("  Search accepted candidates, but final validation did not confirm the improvement.")
    measurement = report.final.get("measurement", report.constants.get("measurement", {}))
    objective = "forward pass"
    if measurement.get("kind") == "library_generation":
        objective = f"library generation ({measurement['generated_tokens']} generated tokens)"
    for w, clocks in report.step_ms.items():
        # the untouched model is re-measured beside the patched one at the end;
        # the job-start clock is a different window and never enters this line
        if clocks.get("speedup"):
            result = (f"{clocks['speedup']:.3f}x confirmed speedup" if clocks.get("win_confirmed")
                      else "no confirmed speedup")
            print(f"  {w}: {objective}: patched {clocks['after']:.3f} ms vs baseline "
                  f"{clocks['baseline_at_end']:.3f} ms ({report.baseline.get('choice')}), "
                  f"measured together: {result} (pair agreement {clocks['stability']:.2f})")
        sequence = report.final.get("sequences", {}).get(w)
        if sequence and clocks.get("sequence_speedup"):
            result = (f"{clocks['sequence_speedup']:.3f}x confirmed speedup"
                      if clocks.get("sequence_win_confirmed") else "no confirmed speedup")
            kind = sequence.get("workload_kind", "repeated_forward")
            description = {
                "repeated_forward": "repeated forward passes",
                "advancing_cache_fixed_tokens": "steps with advancing cache and fixed input tokens",
                "library_generation": "generated tokens via library inference",
            }.get(kind, kind)
            print(f"  {w}: {sequence['steps']} {description}: patched "
                  f"{sequence['candidate_sequence_ms']:.1f} ms vs untouched "
                  f"{sequence['baseline_sequence_ms']:.1f} ms, run whole and alternated: {result}")
    if artifact:
        print(f"artifact: {artifact}")


if __name__ == "__main__":
    sys.exit(main())
