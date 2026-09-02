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
- The plain-vs-compiled baseline choice: the loop times only the plain step, so
  the choice and the fresh-callable rule after a swap are not wired.
- The whole-model check for a reordered-math (assoc-changing) win: the
  region-scale fp32 golden is tested; the model-scale check is not built. A
  quantized decode job needs it.
- Retrace-after-close share updates: the loop drops covered regions but does not
  re-price survivors on the patched model.
- The shape sweep inside a job (gate 7 at swept sizes): locate_span, the store,
  and the child's correctness_only path are tested standalone; the loop does
  not build sweep eval sets, so no job has run gate 7 at a second size.
- The agent mailbox judge (--judge agent) is tested against a thread standing in
  for the agent; no human-or-agent-operated run yet.
- `autotuner/manifest.py` check_build timeout path: never hit.
- Family abandonment and the diminishing-ships / stale-hypotheses close rules: unit
  logic tested via the judge queue tests; no loop run has organically triggered
  them (the scripted judges yield first).
