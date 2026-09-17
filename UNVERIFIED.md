# UNVERIFIED.md

Remaining limits and checks not yet completed. Update entries when there is
new evidence; passing a fixture does not establish model-scale reliability.

September 13 runtime-reduction audit: the broad non-integration suite
(`pytest -m 'not integration' -k 'not live'`) passed 835 tests and failed three
measurement checks. The complete planted-win job also passed with the real
JSON judge protocol and canned responses, including consecutive-step timing
and fresh-process artifact validation. No paid judge or full trained-model
optimization was run for this audit. Regression tests exposed and now cover
scalar/state changes missed by batched identity checks and interrupted deferred
cooling losing its deadline; both were fixed.

The unresolved checks are `test_planted_win_ships`,
`test_injected_slowdown_is_detected`, and `test_correct_kernel_scores_with_the_clock`.
The injected slowdown measured 0.469 ms against a 39.22 ms baseline, with
1.646 ms three-sigma uncertainty. The tiny sandbox operation could not be
separated from its chaining overhead. An isolated replay of the planted kernel
passed correctness and measured a 0.067 ms saving, but its 0.432 ms uncertainty
margin prevented nomination. These results do not establish reliable detection
of small gains or losses. Timing assertions and shipping thresholds remain intact.

September 16 headroom and ranking. Negative step room (FLUX -9%,
RecurrentGemma -25% to -34%) had two causes, both fixed with tests: the
job-start flops probe reads 10 to 25% under what the library achieves at
model shapes (bf16 2.51 vs 3.28 TFLOP/s measured on FLUX's own 4-bit matmul
shape; fp32 2.72 vs about 3.0), so the compute floor sat above the step and
every large matmul region scored no headroom; and the step floor's time term
priced never-evaluated work its flop count skipped (792 GFLOP on
RecurrentGemma, a fifth of the floor). The compute ceiling is now measured in
each pricing window: a 4096-square plain matmul per dtype timed beside the
regions (`RegionPrice.gflops`), the job-start probe raised to 4096 as well,
and peaks rise to any rate the step or a region is measured achieving. The
ranking constant tried first (a 10% headroom floor) is gone. FLUX priced
under the new ceiling (manifest_flux.yaml, one workload, 8.5 min to the first
region): job-start peaks bf16 3360 / fp16 3353 / fp32 2850 GFLOP/s, step room
+18.5% (was -9%), the in-window bf16 rate 3208, and the order is the 52%,
25% and 10% matmul regions first with 12 to 15% estimated headroom each,
where the run of the same morning shipped 5%, 2.2% and 0.8%. Not yet observed
on a live search.

September 16 RecurrentGemma prefill run (`work-2026-09-16-1332`, fp32
checkpoint, prompts of 480/500/520 tokens). Crashed on the first region and
spent 2.5 hours before the search. Three tool bugs, each fixed with a
regression test in `tests/test_regressions.py` and measured on the real model:
(1) a replay wrapper's guard compared a recorded `None` argument with `==`,
which mlx refuses once the argument is an array (the recurrent unit's cache on
the generated-token call); the guard now uses `is None`. (2) The recorder kept
a lazy value snapshot of every array version it recorded, so the scan's 500
in-place writes per layer into a 5 MB array pinned about 46 GB, and the
library's mid-pass eval made it resident: one trace took 109 s at a 59 GB peak
and each of the nine capture passes 113 s, with 3x cooling charged on top. A
trace now keeps no snapshots and a capture keeps only its boundary's: 15.7 s at
15.6 GB and 2.3 s per capture pass. (3) Identity certification blamed the first
mismatching scope by name, not by execution order, so one scope whose compiled
output differed cascaded into 143 fallbacks at one full-model pass each (47
minutes, 81% cooling idle); it now blames the earliest-executed mismatch.
Rerun through the pre-search path with the fixes: 48 minutes to the first
region against 139 (traces 18 s, regions 30 s, settlement 27 min with 54
genuine non-bitwise scopes at one pass each, capture 2 min, pricing 8 min).
Compiling the outermost scopes gains only 1.5% on this fp32 model, and its
compute floor sits above the measured step, so the search itself is expected
to find no kernel headroom; the search has not been rerun.

September 16 widening round and launch-floor stop. The widening round ran
live once (`work-2026-09-16-widening`, Mamba prefill, claude-cli judge, one
region, six attempts): four openers under four kinds against the scaffold,
no refusal, three kernels shipped during search; the end-of-job comparison
then did not confirm a speedup (median ratio 0.998 over 4 pairs). Measured
afterwards (`work-graph-runtime-validation/final-check-attribution-2026-09-16.md`):
the first ship is a real 1.5% win over the compiled baseline, the next two
added nothing (h2 vs h5 and h5 vs h6 both flat), and their confirmed
increments came from timing the live model against fresh copies, which
differ from it by about 1% either way for a GPU-side reason not yet found.
The final check then had a true 1% total to resolve with 4 pairs and could
not. The launch-floor stop has not closed a region in a live run. Replayed over the
September 15 logs it would have closed the Mamba gating region after
attempt 9 in work-2026-09-15-2043 and after attempt 7 in the rerun, with no
later ship lost in either, and never the transpose-matmul region;
whether it ever closes a region that still had a whole-model win in it is
unmeasured. The floor probe swings about 30% between windows with the GPU's
state (6.5 to 13.3 us beside kernels at 11.9 us), which is why three
windows must agree.

The September 6 flow audit completed two bounded CLI runs with a scripted
judge: one rejected its candidate and exported an unchanged model; the other
accepted a deliberately planted optimization, saved a recovery checkpoint,
completed the final model comparison, and verified the artifact in a fresh
process. This verifies the workflow on a fixture, not a new FLUX speedup.
Evidence and the test ledger are in
`work-2026-09-06-flow-audit/audit.md`.

- Graph insertion delivery (2026-09-15). The loop prefers it for every scope
  the screen accepts (`bind/graph.py`); replay stays for the rest, each with a
  logged reason. Verified on the fixture zoo (`tests/test_graph_install.py`,
  `test_multi_shape_bind.py`, `test_sibling_bind.py` on both paths,
  `test_nested_bind.py`, `test_native_fusion.py`, `test_native_search.py`,
  `test_recurrent_cache.py`, `test_inference_flow.py`,
  `test_delivery_accounting.py`) and on small real architectures
  (`tests/test_graph_model_matrix.py`: llama, qwen3.5 and mamba at context 0
  and 5, a FLUX-shaped denoiser for a matmul, attention and an elementwise
  fusion). Identity installs on real Qwen3 0.6B 4-bit, paired against the
  untouched model (`work-graph-runtime-validation/identity-*/results.json`):
  on the final runtime: decode layer -0.2%, decode MLP +1.6%, decode
  attention -1.4% (compiled at its recorded position, bitwise), prefill
  layer +0.3%, prefill attention +0.5% (all unresolved), prefill MLP +1.3%
  (barely resolved); replay identity within noise everywhere; the same
  picture as before the rewrite. On the real Mamba mixer scope one warm wrapper call costs
  80 us of host time, 74 of them MLX's own compiled call (the module handed
  as `inputs=` state is flattened every call) and about 6 the wrapper's; the
  plain module spends 195 us building the same graph (PLATFORM.md,
  compiled-call-host-cost). Python state handed to a scope (a cache) is
  explicit: its arrays are compiled inputs and outputs, its attributes are
  part of the call signature and are written back as the traced call left
  them. Verified on the fixtures: a fresh cache at a recorded position reuses
  the trace and is advanced by the compiled call; a cache with history
  compiles at its recorded position and an unrecorded position runs the
  original module (as a replay variant is guarded to its recorded position,
  so neither delivery serves a position it did not record); a list a call
  appends to is written back at the traced length; llama's block installs by
  graph at context 0 and 5 (`test_recurrent_cache.py`); a graph wrapper runs
  inside the harness's own `mx.compile` of the step. Not verified: a holder
  nested inside an explicit argument (a cache list) through a warm graph
  call, which the runtime walks but no test drives to a write-back (the
  qwen3.5 fixture's block falls back to replay first). Known limits: the
  qwen3.5 fixture's recurrent block is not bitwise the same compiled, which
  identity certification catches before the search and answers with replay;
  a rule replaces every identical occurrence in its scope, so a region
  installed at a subset of its copies there fails bind with a named reason;
  Python settings changed after install are not read by an installed graph
  scope, as a replay wrapper never reads them either. Not yet on a real
  checkpoint: a cache-carrying attention or block scope through graph
  delivery (the Qwen 0.6B generation job records them at three positions;
  whether their compiled identities are bitwise is that job's answer).
- Compilation's own share of a graph install (2026-09-15). Measured on the
  Mamba-370M generation task (`work-graph-runtime-validation/ttft_decomposition.py`,
  results in `ttft-decomposition*/results.json`): the 48 mixer scopes compiled
  with no kernel take the task from 89.7 to 69.0 ms (-23.1%); the shipped
  kernel h3 inside them -23.3%; paired directly, identity against kernel
  -0.0% (3 sigma 0.33 ms); the same kernel through replay -11.0%. Run 3's
  1.328x was compilation's fusion of the mixer, not the kernel's, and the
  region clock (library as plain ops, 1.75x) and the whole-model comparison
  (against the plain model) both credited it to the kernel. The loop now
  plans each region's delivery from the record (`Region.delivery`), runs
  every clock's library arm as the deployed scope will run it (compiled for
  graph delivery: pricing, roofline room, the ladder's ship clock), measures
  a kernel against the incumbent with its new scopes compiled and empty
  (`incumbent_compiled_scopes`), clocks compile alone before the search and
  at each install (`scope_compile`), and splits the final speedup
  (`final.delivery`). Verified on the fixtures
  (`tests/test_delivery_accounting.py`). Then made the default: under
  library inference the manifest's default `baseline: compiled` is realized
  as the model with its outermost compilable scopes compiled and empty,
  installed as the starting state so kernels compose into it and the
  artifact carries it; every whole-model number is against it.
  Verified on the llama fixture (compiled baseline adopted, a kernel composes
  into its wrapper, the artifact exports it). Not yet: a real job with the
  compiled baseline end to end. The earlier accounting under `baseline:
  plain` remains, and was checked on the Mamba-370M generation
  job rerun under it (`work-graph-runtime-validation/mamba-ttft-run4`,
  claude-cli, budget 4/8, 13 minutes): the old winner h3 is `correct_slower`
  against the compiled library arm (0.0101 vs 0.0094 ms per copy), the other
  elementwise region's four kernels win the region clock (+0.002 to +0.003
  ms per copy) and come out inconclusive against the incumbent with the 48
  mixer scopes compiled and empty (ratios 0.996 to 1.004), nothing ships,
  and four per-install clocks (since removed as redundant) put compiling the
  mixers alone at ratio 0.70 to 0.71, resolved. That run's before-search row
  still clocked the innermost scopes (the conv1d children, ratio 1.015,
  inconclusive); the rule is now the outermost certified delivery scopes,
  and delivery is settled by certifying every graph scope's identity before
  any clock. The Qwen3 0.6B 4-bit generation job under the same accounting
  (`work-graph-runtime-validation/qwen-ttft-run1`, 32-token prompt plus one
  token, claude-cli, budget 8, 45 minutes) shipped nothing: every region it
  opened holds a quantized matmul with copies at two shapes (31 tokens and 1
  token), the stitched matmul scaffold matches MLX's prefill dispatch and not
  its one-token dispatch, so eight scaffolds failed the bit-exact smoke gate
  and the judge's one fix each failed the same way (a scaffold gap, not a
  delivery one; a one-token scaffold is the fix). Its before-search row is
  the other finding: the 56 outermost graph scopes (28 MLPs, 28 rope calls)
  compiled and empty made the 29.6 ms task 3.2% slower, resolved: each
  compiled call has a host price and those scopes do not fuse enough to pay
  it. Settling now moves such scopes to replay ("compiling the scope makes
  the step slower"), so a kernel there is measured against plain ops, which
  is what replay delivers. Not yet: a job end to end under that rule
  (the two runs above predate it); whether any kernel beats a compiled Mamba
  mixer (none of eight did); a one-token quantized-matmul scaffold.
- Never-evaluated work (2026-09-15). The recorder writes down every call
  the model makes while building its lazy graph, including results nothing
  returns, keeps, evaluates or reads: under mlx-lm generation the prompt
  call's final norm and output projection are such work (the library drops
  those logits and evaluates the cache alone), so MLX never runs them. The
  maintainer's `work-2026-09-15-2043` job priced that chain as 3.4% of the
  step, opened it as a region with no live outputs, and the ladder child
  crashed indexing an empty output list. Now the record marks such calls
  dead (`Trace.dead`, from a backward pass over what the step returns, keeps,
  evaluates, and every call with an effect), no stretch may contain one, the
  step floor skips them, an evaluated result counts as a live output, the
  ladder refuses a job with no outputs by name, the timed chain names an
  empty pass instead of failing on an index, and the reference-sequence
  fallback answers numeric failures only. In the same pass, a region whose
  ops run in the model's own top-level call (14 on the FLUX denoiser, 3 on
  Qwen prefill) is named unsupported at discovery, since installation swaps
  a module in for its parent and the top level has none; before, every bind
  there failed with "root-scope delivery is not supported" after the budget
  was spent. Verified on a fixture with a dropped and an evaluated branch,
  on the llama fixture under library inference, and on the Mamba-370M
  manifest rerun end to end (`work-2026-09-15-rerun`, the maintainer's
  manifest with `baseline: plain` still in it, claude-cli, budget 15/45,
  47 minutes): 163 ops marked never-evaluated, three regions searched
  past the point of the crash, one kernel shipped through graph delivery,
  the artifact exported and validated. Its accounting reads: kernel against
  the mixers compiled and empty 0.988 (1.2% faster), compiling the mixers
  alone 0.751, total against the plain model 0.742 (1.348x confirmed), so
  the split is 1.33x compilation times 1.03x kernel.
- Library inference measurement (2026-09-15). `use_library_inference` resolves
  before tracing; the job then traces and times mlx-lm's own generation task
  (the prompt plus `final_benchmark.steps` tokens) and every acceptance,
  confirmation, final timing and bundle benchmark uses that same task.
  Verified on small real architectures: llama, qwen3.5, mamba and
  recurrent_gemma match unmodified mlx-lm generation bit for bit with
  independent state per trial (`tests/test_library_inference.py`); the job
  traces and times one task (`tests/test_inference_flow.py`); a bundle with
  a graph-installed kernel loads and runs native generation in a fresh
  process. The first real job (Mamba-370M, prompt 32 + 1 token, claude-cli,
  `work-graph-runtime-validation/mamba-ttft-run`) found the known elementwise
  win but installed nothing: the library task built its cache inside the
  timed call, so the tracer never registered those objects as state holders.
  Fixed: the runtime keeps one cache per layer for its life and resets it in
  place around every trial, and a holder's `state` getter is a recorded state
  call. Run 3 (`mamba-ttft-run3`) then shipped the judge's third edit
  bit-exact through graph delivery on all 48 mixer scopes: 91.5 to 69.1 ms,
  1.328x confirmed, artifact validated in a fresh process; the decomposition
  above then showed that win to be compilation's. Not yet: a transformer
  checkpoint under library inference to an artifact; decode-shaped tasks
  (steps > 1) on a real checkpoint.
- Custom kernel recording at model scale (2026-09-10). A model's own
  `mx.fast.metal_kernel` call now records as one opaque call and a wrapper replays
  it by import path. Verified on the fixture zoo (trace, replay, region barrier,
  identity-certified wrapper, fp32 refusal) and on tiny random Qwen3.5, BitNet,
  and Mamba2 models. No full job has run over a model with a custom kernel, and
  no artifact carrying such a wrapper has been applied in a fresh process; the
  first Qwen3.5-4B-4bit run (3 GB download) is the check.
- Timing sensitivity remains unresolved. The audit's final focused run passed
  51 tests but failed two hardware checks: detecting a planted 3% slowdown and
  reproducing a tiny region's share across two measurements. In the slowdown
  check, the measured slowdown was 1.60 ms while the three-sigma uncertainty
  was 4.88 ms, so the clock correctly refused a confident decision. This does
  not establish that it can resolve a 3% change reliably. The pricing check
  produced shares of 32.7% and 1.42% on a tiny fixture. CPU load and a bandwidth
  probe did not catch these conditions. Ranking small regions can therefore
  be unreliable even when preflight passes. See
  `work-2026-09-06-flow-audit/remaining-checks.log`. Assertions and acceptance
  thresholds were not relaxed to hide these failures.

- A win on trained model weights. The supplied FLUX architecture with fixed
  random weights now has a validated 3.36% forward-pass latency reduction and
  an artifact checked in a fresh process. See
  `work-2026-09-04-refinement/flux-saved-check/report.json`.
  That saved result was not independently re-timed in the September 6 audit.
  Trained FLUX weights have not been checked. Five earlier 8B jobs ran (the
  work-2026-08-31-* folders): tracing, regions, capture, pricing, ranking,
  peaks, both step clocks, artifact emit, and fresh-process apply() are
  exercised there. The later runs also reached a live claude-cli judge (27
  calls in the decode run), put judge-written kernels through the ladder (14
  measured in the 22:54 run), and had bind's identity certification refuse a
  scope. None of those 8B jobs shipped a region.
- Completing the current search loop at model scale: the Claude run in
  `work-2026-09-05-2000-est` accepted three successive improvements to one
  FLUX region, each with a completed whole-model comparison, then failed on
  a candidate in the second region before final validation or export. The
  candidate's inverted fallback condition and the lost worker error detail
  are fixed and covered by regression tests. No full FLUX CLI run has
  completed after the September 6 audit changes. Opaque compiled calls
  replayed by import path and dependent-matmul chains also still need
  model-scale search coverage beyond fixtures.
- The deployment-shaped final timing (whole runs of consecutive forward
  passes, original and patched alternated in both orders, cooling between
  runs) has run on FLUX outside a job, three times. Two runs that started
  right after heavy GPU work read 1,850 ms per step throughout (a hot start;
  PLATFORM.md) and gave 1.072x and 1.074x; a run in a quiet process read
  1,220 ms per step, matching the job's paced clock, and there the kernels
  gained 1.106x over ten-step runs, the same as the paced 1.10x; the
  wrappers cost nothing resolvable in either protocol (PLATFORM.md). The protocol's A/A null
  in that quiet process resolved a 0.04% difference between identical arms
  as a win, so the sequence decision has no measured null under it yet. No
  job has completed its own final sequence check at FLUX scale; fixtures
  only (`tests/test_install.py`, `tests/test_multi_shape_bind.py`).
- The artifact bundle's `load.py`, `validate.py` and `benchmark.py` have run
  on fixtures only (`tests/test_artifact.py`, plus the install and multi-shape
  tests that export through the loop). No FLUX-scale artifact has been built
  from its own `model/` copy or checked by the bundle's scripts; the
  2026-09-06 FLUX artifact predates the bundle and has no `load.py`.
- Installing a parent-scope kernel over a module whose child already holds
  one (the FLUX feed-forward shape: linear_in patched, then the
  SwiGLU-multiply plus linear_out cut) failed every time in the 2026-09-06
  run, because the parent's replay was emitted with only its own cut and ran
  the library where the child's kernel belonged. The composed replay
  (`compose_scope_variants`) fixes it. Verified on the real FLUX model with
  the run's saved kernels by replaying the run's install history, including
  the second single-block kernel over the first, rejected re-installs on
  every occupied scope, and the real whole-model check at two pairs
  (scratchpad script, three passes), and by `tests/test_sibling_bind.py` in
  both install orders. No full FLUX search has run through this path.
- Root-scope delivery (a region whose only scope is the top-level callable, e.g. a
  plain-function model): the loop rejects it with a named reason; apply() handles a
  "" scope but no run has exercised it.
- Completing a compiled-baseline model search: FLUX preparation on 2026-09-05
  measured the plain forward pass at 1,343.7 ms and the compiled pass at
  1,289.2 ms, then priced 22 regions against the compiled baseline. See
  `work-2026-09-04-refinement/flux-preparation/run.jsonl`. This exercised a
  model-scale compiled baseline with fixed random weights. The later saved
  candidate check exercised scoring, installation at 20 copies, a whole-model
  win, and fresh-process artifact validation against that compiled baseline.
  Stateful decode jobs still take the plain baseline when the trace shows
  retained state. Choosing the faster baseline automatically is not built;
  the manifest decides.
- Context workloads (`context: N` on a workload; the harness builds, fills and
  rewinds the model's KV cache around the plain model file): verified on the
  fixture `tests/fixtures/context_model.py` end to end (trace, bind, retrace,
  the whole-model check, the artifact rebuilt in a fresh process through both
  apply() and the bundle's load()), and on the real Qwen3 0.6B 4-bit decode
  step against the retired model-file wrapper, in separate processes
  (2026-09-07): the same 734 recorded ops, 28 state calls, 64 regions, the
  plain baseline for the same reason, the same cache buffer and offsets,
  repeatable calls; only the root scope's attribute name differs (`inner`
  became `model`). Not yet run on a real checkpoint: a full job through the
  ladder to an artifact with a context, and `manifest_llama_decode.yaml`
  itself (Llama 3 8B, a 5 GB checkpoint).
- Live cache-offset replay (the "rope fix"): a scalar the model reads off a
  KV cache (rope's position) replays as that read, not the value recorded.
  The recorder snapshots the cache offset before each state call; the emitter
  rebinds an ``offset`` kwarg whose value is that cache's offset, structurally
  (the cache updated in a scope enclosing the call), to a live ``<cache>.offset``
  read. Verified: fixture tests/test_rope_offset.py (the wrapper reads the live
  offset; a real advancing decode stays bit-identical to the unwrapped model at
  offsets 512-515 where a baked-512 wrapper drifts 0.30->1.30; a no-context
  prefill keeps the literal 0), and on real Qwen3 0.6B 4-bit (layer-0 self_attn
  wrapper, advancing decode, live bit-identical vs baked drifting) and Llama 3
  8B 4-bit (the emitted attention wrapper reads the live offset, no baked 512).
  Bit-identical at the recorded position, so certification and retrace do not
  regress. Not yet exercised: a full decode job shipping a kernel whose wrapper
  scope spans rope, then deploying that artifact in a real generation loop.
- The whole-model check for a reordered-math (assoc-changing) win is built.
  Its streamed fp32 reference, including quantized operations and dtype casts,
  has run on the small fixture in
  `tests/test_regressions.py::test_whole_model_golden_promotes_casts_and_quantized_math`.
  Building that reference and accepting a reordered-math win on FLUX or an 8B
  model remain unverified.

- The whole-model fp32 reference for reordering ("changing") kernels is now
  the untouched model itself run at fp32 (`ladder.golden.fp32_reference`),
  not a replay of its recording: float parameters, module buffers, inputs
  and every state object's arrays are lifted for the call and restored;
  packed quantized weights stay packed and unpack on the fly; the run is
  recorded and refused if any op hands back a lower precision. Nothing
  sealed can stop it, so a KV cache or a pre-compiled activation no longer
  blocks grading. Verified (2026-09-08): agrees with the recording-replay
  engine to 1e-5 on a stateless model in bf16, fp16 and fp32; keeps 4-bit
  weights packed and matches dequantize-then-matmul; refuses a forward that
  hard-casts to bf16; runs a stateful step leaving the live cache untouched
  and repeatable; deterministic. On the real bf16 Qwen3 0.6B decode step the
  reference builds where the 00:14 run crashed, and the run's own reordering
  kernel (r17bdaa_vec_ilp) binds, grades at 1.005x the library's error, and
  ships at all 28 scopes. On FLUX.2 4B (1682 ops, 35 compiled sections,
  4-bit) the same unchanged builder costs 3.8 GB extra, not 16. NOT yet: a
  full judge run to a shipped artifact with a reordering kernel.
- The region clock's blindness to overlap (PLATFORM.md overlap-is-the-mechanism)
  is measured, not fixed: a ship clock that times a region's copies in their
  model structure (independent siblings a layer runs concurrently) rather
  than one dependent launch at a time is not built, and neither is a
  whole-model comparison with enough pairs to resolve a ~1% win. Whether
  the same gap explains the 2026-09-03 Qwen 0.6B decode refusals is inferred
  from the matching signature, not re-measured.
- The whole-model changing check's error metric was a per-element relative
  error with the denominator clamped at 1e-6, which explodes wherever the
  reference crosses zero; on a 151,936-logit output it read 1439 vs 3981
  (a 2.77x "regression") for a kernel every scale-aware metric puts at 1.01x
  the library, same argmax token. It had never run on a real model.
  `e2e._scale_error` now floors the denominator at each output's own
  magnitude (the region gate's stated intent, "epsilon at observed scale").
  The region gate (child.py) still uses the raw metric behind its measured
  k-set floor; that floor absorbs the artifact (it read 3.76 in the run) but
  the metric there is the same and is a follow-up.
- Survivor share updates after installation are implemented: covered members
  are removed and measured survivors are repriced against the patched model.
  The September 5 Claude run reached the next region after three promotions.
  The September 6 audit also added recapture when trimming changes a cached
  region's representative. That defensive branch is tested with a scripted
  queue; it has not been observed in a real run and was not the Claude crash's
  cause.
- The shape sweep at model scale: a job now traces and captures every named
  dim at its sweep sizes, gate 7 checks each kernel there, and the final check
  runs the patched model at each size. Tested on the planted-win fixture only;
  no 8B job has run with a named dim since.
- The agent mailbox judge (--judge agent) is tested against a thread standing in
  for the agent; no human-or-agent-operated run yet.
- `autotuner/manifest.py` check_build timeout path: never hit.
- Budget-only persistence with a live judge: refused yields, refused plan
  edits, and the attempts they cost are tested against scripted judges only.
- A region reading a state call's returned view, such as attention reading a
  cache slice, may pay a contiguity copy in the wrapper that the region clock
  did not price. Whole-model timing remains the acceptance check. Dictionary
  weight lookup is now fixed and verified by a traced wrapper test, including
  identifier keys and nested lists; model-scale use still needs coverage.
- Per-workload specialization is incomplete. The current promotion rule
  requires a resolved whole-model win on every declared workload, even a
  workload where the region does not occur. This can reject useful targeted
  improvements. Shape-specific wrappers and correctness fallbacks are tested,
  but they do not implement the design's full specialized search policy.
- Recovery checkpoints preserve accepted code and its completed measurements.
  They are explicitly pending final validation and fresh-process validation.
  There is no automatic resume command, and an interrupted checkpoint has not
  been recovered and revalidated at FLUX scale.
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
- The measured floor inside candidate scoring at model scale: the 2026-09-05
  FLUX preparation ran the stream probe beside quantized projections with
  packed uint32 weights, scales, and biases during region pricing. Its first
  shortlist took 412.2 seconds, including 296.3 seconds of cooling; see
  `work-2026-09-04-refinement/flux-preparation/prepared.json`. The candidate's
  grouped library/candidate/probe clock subsequently ran during the Claude
  search. These floor estimates can be beaten by measured kernels; they are
  search guidance, not proven physical limits or acceptance gates.
- The flops term of the roofline is still arithmetic against the matmul peak
  measured at job start, so a compute-bound region's headroom mixes a probe
  from one window with a peak from another. Decode regions are memory-bound
  and never touch it; a prefill job would.
- A quantized_matmul whose scales and biases are float32 while x is bfloat16
  (FLUX emits 2 of its 89 quantized matmuls): both stitches refuse it (scales
  dtype != x dtype) and the naive lowering refuses it (mixes bf16 and f32), so
  it has no starting kernel and its region is skipped. The library's own
  mixed-input path is not reproduced; whether those two regions are worth
  covering depends on how they price, which no job has measured.
- GPU command cancellation after a worker is killed is not guaranteed. The
  2026-09-05 WindowServer crash occurred while a FLUX candidate was being
  evaluated; see `work-2026-09-04-refinement/crash-review.md`. The parent now
  watches five-second GPU evaluation windows in candidate workers, including
  first-use JIT, and stops the job on timeout. CPU worker tests verify
  supervision and separate cooling from that deadline. Parent-process model
  work is not fully isolated from driver hangs. These tests do not verify
  cancellation of stalled GPU commands or prove that desktop crashes are
  impossible.
