# UNVERIFIED.md

Every code path not yet executed on real hardware, and why. Delete entries as they run.

- A win at model scale. Five 8B jobs have run on this machine (the
  work-2026-08-31-* folders): tracing, regions, capture, pricing, ranking,
  peaks, both step clocks, artifact emit, and fresh-process apply() are
  exercised there. The later runs also reached a live claude-cli judge (27
  calls in the decode run), put judge-written kernels through the ladder (14
  measured in the 22:54 run), and had bind's identity certification refuse a
  scope. No job has shipped a region, so a real ship, bind's install, e2e,
  rollback, and an artifact with a non-empty swap table have run only on the
  planted-win fixture.
- Everything the 2026-09-01 refactor changed, at model scale: the verdict-first
  cycle with a live judge, the second-win rule, the stress regime that steps its
  magnitude down, the fresh-process artifact check, opaque compiled calls
  replayed by import path, and the dependent-matmul chain rule. All are tested
  on fixtures only. The next 8B run is their test.
- Root-scope delivery (a region whose only scope is the top-level callable, e.g. a
  plain-function model): the loop rejects it with a named reason; apply() handles a
  "" scope but no run has exercised it.
- The compiled baseline at model scale: every clock now runs against the
  manifest's baseline (compiled by default: step clocks, pricing, the child's
  library arm, the veto, the headline), verified on fixtures and pinned by
  platform tests. No 8B or Qwen job has run under it. Choosing the baseline
  by measurement is not built; the manifest decides.
- `models/qwen3_0.6b_decode.py`: the decode wrapper is verified on a tiny
  random Qwen3; the real build() (a 1.2 GB download on first use) has not run.
- The whole-model check for a reordered-math (assoc-changing) win: the
  region-scale fp32 golden is tested; the model-scale check is not built. A
  quantized decode job needs it.
- Retrace-after-close share updates: the loop drops covered regions but does not
  re-price survivors on the patched model.
- The shape sweep at model scale: a job now traces and captures every named
  dim at its sweep sizes, gate 7 checks each kernel there, and the final check
  runs the patched model at each size. Tested on the planted-win fixture only;
  no 8B job has run with a named dim since.
- The agent mailbox judge (--judge agent) is tested against a thread standing in
  for the agent; no human-or-agent-operated run yet.
- `autotuner/manifest.py` check_build timeout path: never hit.
- Family abandonment and the diminishing-ships / stale-hypotheses close rules: unit
  logic tested via the judge queue tests; no loop run has organically triggered
  them (the scripted judges yield first).
