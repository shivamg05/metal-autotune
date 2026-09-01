# UNVERIFIED.md

Every code path not yet executed on real hardware, and why. Delete entries as they run.

- The search half at model scale: a full 8B job now ran end to end (2026-08-31,
  work-2026-08-31-0of22/RUN_FINDINGS.md): trace, regions, capture, pricing, ranking, the region
  open/skip/close loop, peaks, both step clocks, artifact emit, and fresh-process
  apply() are all exercised on real hardware. But every ranked region closed at
  the scaffold step (quantized_matmul, fast.rope, fast.sdpa are not lowerable),
  so the judge, the ladder, bind, e2e, and rollback have still never run at model
  scale, and no live-judge call has happened inside a job. Scaffold coverage for
  quantized_matmul (stitched from the wheel's own MSL, with naive fallback) and
  fast.rope is now built and verified against the real 8B regions standalone;
  16 of 17 viable decode regions are scaffold-buildable. What remains unverified
  is the composition: a stitched or lowered scaffold climbing the full ladder
  inside a job with a live judge. The next decode run is that test.
- Root-scope delivery (a region whose only scope is the top-level callable, e.g. a
  plain-function model): the loop rejects it with a named reason; apply() handles a
  "" scope but no run has exercised it.
- Compiled-baseline path (plan section 2): the loop times only the plain step; the
  plain-vs-compiled choice and law 10's fresh-closure rule are not wired.
- Model-scale golden for the e2e assoc-changing path (plan 7.11): region-scale
  golden is tested; the promotion interceptor is not built. The M12 flagship
  (quantized decode) needs it.
- Retrace-after-close share updates: the loop drops covered regions but does not
  re-price survivors on the patched model (plan section 11, second bullet).
- Sweep-instance capture inside the loop (gate 7 at swept sizes): locate_span, the
  store, and the worker's correctness_only path are all tested standalone; no loop
  run with a named-dim manifest has exercised the wiring end to end.
- A full live-judge job: no complete job has run with a real model as judge. The
  transports themselves are verified: the API client has its env-gated contract
  test (ANTHROPIC_LIVE_TEST=1, needs a key), and the claude-cli transport passed
  a real seed call on this machine's Claude Code login (CLAUDE_CLI_LIVE_TEST=1,
  2026-08-30). The agent mailbox transport is tested against a thread standing in
  for the agent; no human-or-agent-operated run yet.
- `autotuner/manifest.py` check_build timeout path: never hit.
- Family abandonment and the diminishing-ships / stale-hypotheses close rules: unit
  logic tested via the judge queue tests; no loop run has organically triggered
  them (the scripted judges yield first).
