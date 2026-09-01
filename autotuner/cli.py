"""autotune run manifest.yaml"""

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="autotune")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one optimization job")
    run.add_argument("manifest")
    run.add_argument("--work-dir", default="autotune_work")
    run.add_argument("--artifact", default="artifact")
    run.add_argument("--model", default="claude-opus-5", help="judge model id")
    run.add_argument("--judge", choices=("api", "claude-cli", "agent"), default="api",
                     help="api: Anthropic SDK (needs ANTHROPIC_API_KEY); "
                          "claude-cli: local `claude -p` on this machine's Claude Code login; "
                          "agent: a live agent answers <work-dir>/judge_io (see AGENT_JUDGE.md)")
    args = parser.parse_args(argv)

    from .log import fmt_signed_ms
    from .loop import JobRunner

    if args.judge == "api":
        from .judge.client import AnthropicJudge
        judge = AnthropicJudge(model=args.model)
    elif args.judge == "claude-cli":
        from .judge.agent import ClaudeCLIJudge
        judge = ClaudeCLIJudge(model=args.model)
    else:
        from .judge.agent import AgentFileJudge
        mailbox = Path(args.work_dir) / "judge_io"
        judge = AgentFileJudge(mailbox)
        print(f"agent judge: answer requests in {mailbox}/ (protocol in AGENT_JUDGE.md)")

    judge.transcript = Path(args.work_dir) / "judge.jsonl"
    print(f"judge: {args.judge} ({args.model})")
    print(f"run log: {Path(args.work_dir) / 'run.jsonl'} (one JSON line per event; tail it)")
    print(f"candidates: {Path(args.work_dir) / 'candidates.log'} (one line per attempt)")
    runner = JobRunner(
        args.manifest,
        args.work_dir,
        judge_factory=lambda region: judge,
    )
    report = runner.run()
    artifact = runner.emit_artifact(args.artifact)
    shipped = [r for r in report.regions if r.get("s")]
    print(f"job done: {len(shipped)}/{len(report.regions)} regions shipped")
    for w, clocks in report.step_ms.items():
        before, after = clocks.get("before"), clocks.get("after")
        if before and after:
            print(f"  {w}: {before:.3f} ms -> {after:.3f} ms "
                  f"({fmt_signed_ms(before - after)}, {(after - before) / before * 100:+.1f}%)")
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
