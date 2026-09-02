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
- The compiled baseline at model scale: every clock runs against the job's
  baseline (compiled by default: step clocks, pricing, the child's library
  arm, the veto, the headline), verified on fixtures and pinned by platform
  tests. A decode step keeps a KV cache, so both decode jobs take the plain
  baseline by the harness's own rule (the 2026-09-01 23:48 Qwen run found the
  compiled call killing the model; fixed by detection from the trace). The
  compiled baseline has run only on stateless fixtures. Choosing the baseline
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
- Budget-only persistence with a live judge: refused yields, refused plan
  edits, and the attempts they cost are tested against scripted judges only.
- Two known edges of delivery, neither exercised by a real model yet: a
  weight or state object stored under a plain dict with an identifier key is
  reached by attribute in the generated wrapper and fails at certification
  with an AttributeError rather than a named reason; and a region that reads
  what a state call returns (attention reading the cache slice) receives a
  view, whose contiguity copy at the kernel call the region clock never priced.
- State calls at model scale: the Qwen3 4-bit decode step traces with 28
  `state:KVCache.update_and_fetch` calls and 66 of 73 candidates pass the
  scope screen, and the attention and block scopes certify on the fixture; no
  job has yet certified the real attention scope, shipped a kernel through it,
  or loaded such an artifact in a fresh process.
- Starting kernels for the chains state calls made reachable: on the real model
  the naive kernels for norm+QKV, residual+norm+gate+up, down+residual+norm, and
  gate+up build, match the library to bf16 rounding, and read 4 to 6x the
  library on a shared GPU; none has been through the ladder. Attention itself,
  dequantize, and slice reads still have no starting kernel.
- The measured floor at model scale: the stream probe has run beside bf16
  matvecs and fixture chains, not yet beside a quantized projection (uint32
  weights, scales and biases as separate inputs) or inside a real job's
  pricing and sandbox clocks. The next 4-bit Qwen3 run is its test.
- The flops term of the roofline is still arithmetic against the matmul peak
  measured at job start, so a compute-bound region's headroom mixes a probe
  from one window with a peak from another. Decode regions are memory-bound
  and never touch it; a prefill job would.

