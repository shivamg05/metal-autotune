# Implementation Plan: Autonomous MLX Kernel Optimization

This plan turns `DESIGN_SPEC.html` into a working tool. Read the spec end to end first.
The spec is the authority on WHAT the system does and its vocabulary (manifest, workloads,
trace, regions, roofline, scaffold, hypothesis queue, the ladder, bind, the artifact).
This plan is the authority on HOW to build it: the stack, the module boundaries, the data
formats, the build order, the platform facts you must verify before trusting them, and the
decisions the spec leaves open. Where this plan and the spec disagree, the spec wins.

Everything here assumes a fresh repository containing only the spec and this plan.

---

## 1. The system in one paragraph

An MLX model runs on a Mac. Every math op it calls is a Metal GPU kernel. We record the
exact op-call stream the model runs on declared workloads, cut that stream into regions
(runs of consecutive op calls that one kernel could replace), and price each region by
what fusing and specializing it could physically save. Then a loop opens the best region:
the harness builds a correct starting kernel, an LLM (the judge) proposes one small edit
at a time, and the harness compiles, checks correctness, and times each edit out of
process. A kernel that is correct and faster on the region clock gets bound into the
model; a retrace and an end-to-end check confirm the bind is real; only then it ships.
The job ends with an artifact (kernels, a bind table, an `apply()` installer, a report)
that speeds up a freshly loaded model without any of the search machinery.

Three components, kept strictly apart:

- **The harness**: ordinary code. Owns tracing, region cutting, pricing, all measurement,
  all correctness gates, bind, and the artifact. It never judges a kernel by taste, only
  by measurement.
- **The judge**: an LLM behind a narrow JSON boundary. Sees metadata only (shapes, dtypes,
  verdicts, roofline distance, kernel source it is editing). Never sees tensors, weights,
  or activations. Never overrides a failed check. Proposes one hypothesis at a time and
  writes one Metal kernel body for the front item of its queue.
- **The artifact runtime**: a slim, dependency-light package that ships inside the
  artifact. It contains the bind wrapper and `apply()`. The harness uses the same wrapper,
  so the thing you measure is the thing you ship.

Hard laws, repeated because everything else bends around them: dtypes and quantization
are frozen. Faster-but-wrong is discarded. Correct-but-slower never ships. The judge
never grades its own work.

---

## 2. Stack and ground rules

- Python 3.12, `uv`-managed venv. Dependencies: `mlx` (pin the exact version in
  `pyproject.toml` and record it in every report; this plan's platform facts were
  verified on mlx 0.32.2), `pyyaml`, `pytest`, `anthropic` (judge client only).
- Target: Apple Silicon Macs. Nothing may assume a specific chip; peaks are measured
  per job, never read from a table.
- The harness never adds `mx.compile` to anything. The baseline is the model exactly as
  `build()` hands it, run plain. (Motivating history: `mx.compile` measured ~4% slower
  than plain on a small test net. An M0 spike re-checks this on the pinned version; if
  it ever flips, the rule still stands, because the spec defines the baseline as the
  model as handed. If the model's own code compiles internally, that is the model's
  business and the tracer treats compiled sections as opaque calls.)
- No GPU-dependent logic gets a CPU fallback path. If the environment cannot measure,
  raise; a fallback that "runs while proving nothing" is worse than a crash.
- Prior-measurement claims in this plan come in two kinds. Claims the build RELIES on
  are marked "re-verify" and each has a Milestone 0 spike. Claims that only motivate a
  rule are marked "motivating history"; their numbers are illustrative and nothing may
  be derived from them.
- Keep an `UNVERIFIED.md` at the repo root from day one: every code path that has not
  yet executed on real hardware, and why. Delete entries as they run. The file shrinking
  to nothing is a milestone.

---

## 3. Repository layout

One installable package plus a deliberately separate runtime package. Names are
suggestions; the boundaries are not.

```
autotuner/
  manifest.py        # parse + validate + default the manifest; frozen job config
  workload.py        # materialize input tensors from specs; sweep sizes; seeds
  trace/
    patches.py       # the patch sets: mx.* functions, array dunders/methods,
                     # per-subclass Module __call__, top-level callable, mx.compile
    recorder.py      # record mode: node capture, array identity tracking
    freeze.py        # edges, liveness, completeness check, drop array refs
    replay.py        # re-execute a recorded node sequence (Section 5.4)
  measure/
    session.py       # duty-cycle pacing, session health log, warm-until-stable
    clocks.py        # step clock, region clock, paired interleaved comparison
    peaks.py         # bandwidth + per-dtype flops microbench, running-max estimator
  regions/
    build.py         # singletons + chain growth + rejection rules + per-stretch liveness
    fingerprint.py   # copy grouping: canonical region identity
    price.py         # boundary tensor capture, region clock share, the floor
    roofline.py      # boundary bytes (region inputs + outputs ONLY, never
                     # intermediates), flops summed over ops, launch count;
                     # T_mem/T_compute/T_launch, bound, s_max
    rank.py          # ordering, tie-breaks, overlap bookkeeping
  scaffold/
    lower.py         # naive lowering: region ops -> one correct Metal kernel
    stitch.py        # MSL include flattener + stitching from the shipped kernel tree
  ladder/
    static_checks.py # gate 1: IO contract, boundary dtypes, live outputs, fallback decl
    gates.py         # gates 2..9 orchestration, per-gate structured results
    golden.py        # fp32 golden evaluator for assoc-changing compares
    numeric.py       # tolerance compare, non-finite pattern rules, value regimes
  sandbox/
    worker.py        # subprocess entry point: runs requested gates on one kernel
    protocol.py      # JSON request/response schema, timeouts, kill handling
    poison.py        # init_value poisoning + pool saturation
  judge/
    schema.py        # hypothesis, queue item, verdict record types, launch grammar
    prompts.py       # what the judge sees, rendered from region state
    client.py        # Anthropic API calls, strict JSON in/out
    scripted.py      # deterministic fake judge for tests
  bind/
    wrapper.py       # re-export of autotuner_runtime.wrapper (implementation lives there)
    verify.py        # retrace dataflow check (Section 7.10)
  e2e.py             # orig-vs-orig floor, patched-vs-original (both assoc paths),
                     # step-time veto (Section 7.11)
  loop.py            # region open/close state machine, close rules, budgets
  artifact/
    emit.py          # write kernels/, bind_table, report; package the runtime
  report.py          # report.json schema and writers
  log.py             # append-only run log, sign conventions
  cli.py             # `autotune run manifest.yaml`

autotuner_runtime/   # separate package, no imports from autotuner/
  wrapper.py         # THE bind wrapper implementation: slim intercept layer,
                     # instance-keyed arming, substitution map, step delimitation
  apply.py           # load bind_table + kernels, resolve addresses to instances on a
                     # fresh build() model, install the wrapper

tests/
  fixtures/          # toy model zoo (Section 12)
  cheats/            # adversarial kernel zoo (Section 12)
  test_*.py
spikes/              # runnable spike scripts from Milestone 0; results -> PLATFORM.md
PLATFORM.md          # verified platform facts, each with the spike that proves it
UNVERIFIED.md
```

Dependency direction: `judge/` may import `regions/` types to render prompts; nothing
else imports `judge/` except `loop.py`. `sandbox/worker.py` imports only what one
evaluation needs. `autotuner_runtime` imports nothing from `autotuner`.

---

## 4. Data model

Define these as typed records early (dataclasses serialized to JSON; tensors to
safetensors via `mx.save_safetensors`). Nearly every module boundary is one of these
crossing it.

**Manifest** (YAML in, frozen dataclass out). Fields per the spec: `model` (path to a
file defining `build()`), `workloads` (list of `{inputs: [{shape, dtype}], name?}`),
`sweep` (named dims to sizes), `tolerances` ({rtol, atol}), `budget` ({per_region,
total}). Validation: `build()` must exist, take no args, and be importable in a fresh
subprocess. Named dims are strings inside shapes; integer dims are constants and never
swept. Every defaulted field is recorded in the report with its default.

**TraceNode**: `{seq: int, op: str, in_arrays: [array_id], out_arrays: [array_id],
in_specs: [(shape, dtype)], out_specs: [(shape, dtype)], scalar_args: {...},
module_address: str, position_in_module: int}`. `array_id` is a stable identity the
recorder assigns while it holds every array alive for the pass. `module_address` is the
module instance path plus call index (two calls of the same layer are distinct
addresses); the traced top-level callable is itself an address (the empty path plus call
index), so a model that is a plain function still yields installable addresses for its
ops. The (module_address, position_in_module) pair is the install address for a wrapper.
`position_in_module` counts recorded calls per op name within the address, so the pair
survives partial patch surfaces.

**Trace** (per workload, per traced size): `{nodes: [TraceNode], edges: producer ->
consumers, step_outputs: [array_id], weights: [array_id], inputs: [array_id],
liveness: array_id -> {consumed_by: [seq]} | python_retained | step_output}`. Note
there is no "dies inside" tag at freeze time: whether a value dies inside a stretch is a
per-stretch derivation `regions/build.py` makes from `consumed_by` against the stretch's
span. Freezing runs the completeness check from the spec: every array feeding a node is
a workload input, a weight, or an earlier node's output, and every step output comes
from a node, or the trace aborts naming the offending call.

**Region**: the spec's record: `{ops: [TraceNode refs], copies: int, inputs, outputs
(live values included), workloads: [name], p, T_orig_ms, bound: memory|compute|launch,
roofline: {T_mem, T_compute, T_launch, T_roofline, s_max}}`, plus `fingerprint`
(Section 5.5), `sweep_instances` (per sweep size: the matched node span, shapes, and
scalar_args at that size; Section 5.6), and `boundary_store` (paths to saved
input/output tensor sets, k per workload; Section 5.7).

**Hypothesis / queue item**: `{id, kind, assoc_tag: preserving|changing,
family_id: str, hypothesis: str, depends_on: id?, condition: correct|shipped|failed}`.
`family_id` names the algorithm family the item belongs to (one-pass vs two-pass
attention, split-K, ...); the judge declares it. When omitted, an item inherits its
parent kernel's family, and a parentless item starts a new family, so unrelated
hypotheses are never lumped into one family's counters. The harness keeps the 8-strike
abandonment counter and climb state per `family_id`; head and shipped remain the spec's
two per-region bookmarks (never per-family), and abandoning a family resets head to the
scaffold or the last shipped kernel, exactly as the spec says. Only the front ready
item is ever turned into Metal.

**KernelAttempt**: `{hypothesis_id, parent_kernel_id, source: str, header: str,
launch: {grid, threadgroup, template}, fallback_predicate?: str}`. `grid`,
`threadgroup`, and `fallback_predicate` are expressions in the launch grammar
(Section 5.8), evaluated by the harness against actual shapes at every call, so one
kernel can launch correctly at every sweep size. Every attempt is stored whether it
fails or ships; failed kernels are legal parents.

**Verdict**: `{hypothesis_id, outcome: failed|correct_slower|tentative_ship|shipped|
rolled_back, failed_gate?: str, gate_detail?: {...}, region_ms?: float,
samples?: [...], e2e?: {...}}`.

**BindEntry**: `{region_fingerprint, kernel_id, launch,
copies: [{member_addresses: [(module_address, position)],
arm_address: (module_address, position),
input_map: [kernel_input_role -> (member_address, position, arg_index)],
substitutions: [(member_output_ref, kernel_output_index)]}],
shape_dispatch: [(shape_predicate, kernel_id)], fallback: library}`.
Each copy of the region has its own member set, its own arm point (the last op of that
copy's sequence), and its own input map saying which argument of which member call
supplies each kernel input. Addresses are serialization only: at install time (harness
bind or artifact `apply()`), addresses are resolved against the concrete model to module
instances, and the wrapper dispatches on instance identity (Section 5.3).

**Artifact** on disk, per the spec:

```
artifact/
  kernels/<id>.metal + <id>.launch.json
  bind_table.json          # [BindEntry]
  buffers/                 # reserved, empty in v1 (future precompute)
  runtime/                 # the autotuner_runtime package, vendored
  apply.py                 # thin shim calling runtime.apply
  report.json
```

---

## 5. Core mechanisms, decided

These are the decisions the spec leaves open, made now so the implementation does not
stall on them. Each records why, and the risky ones have Milestone 0 spikes.

### 5.1 Tracing mechanics

- Patch surface: every array-returning callable in `mx`, `mx.fast`, `mx.linalg`,
  `mx.random`, `mx.fft`, plus every operator dunder and method on `mx.array`, plus
  module calls (next bullet), plus the top-level model callable, plus `mx.compile`.
  Verified on mlx 0.32.2: module-level `setattr` on the compiled extension module works
  and `mlx.nn` layers see it (they look ops up at call time); `mx.array` is a nanobind
  heap type, its dunders accept `setattr`, and CPython propagates the patch to the C
  slots, so `a + b`, `a @ b`, slicing, and `x[i] = v` are all interceptable. Patching
  `mx.add` alone does NOT catch `a + b`; reflected operators (`__radd__` etc.) need
  their own patches. Re-verify all of this on the pinned mlx version in Milestone 0; it
  is the wager the whole design rests on.
- Module calls: `mlx.nn.Module` does NOT define `__call__`; every layer subclass
  defines its own (verified on 0.32.2), so patching the base class intercepts nothing.
  Instead, after `build()` returns, walk the model's module tree and patch
  `type(m).__call__` once per distinct subclass encountered, dispatching on `self`
  identity for addressing. Per-instance attribute assignment does not work because
  dunder lookup bypasses instance dicts. The top-level callable (whatever `build()`
  returned) is wrapped directly; it delimits the step, roots the address space, and
  hosts the counter resets.
- The recorder holds a strong reference to every array seen during pass 1 (the pass is
  lazy, so normally no tensor memory is materialized). Identity is `id()` plus that
  liveness guarantee, mapped to stable `array_id`s at freeze. Caveat: a model that
  calls `.item()` or `mx.eval` internally (data-dependent branches, which the spec
  embraces) evaluates mid-record, and the recorder's strong references then keep
  materialized intermediates resident for the whole pass. Detect in-pass evaluation and
  log a memory warning; the eager fixture in M3 pins the behavior.
- Liveness of produced arrays, decided at freeze time: `consumed_by` (later recorded
  calls that read it), `step_output` (in the returned tree), and `python_retained`.
  Retention is detected two ways, both applied: (a) drop the recorder's references,
  force a GC pass, and probe weak references (verified working on 0.32.2: `mx.array`
  accepts weakrefs and dies on schedule); (b) positively, snapshot every array reachable
  from the model object's own attributes/pytree after the pass and mark anything found
  there `python_retained` regardless of recorded consumers. The positive check matters
  because retained-AND-consumed is the dangerous case: a KV append (`cache.append(k)`
  then `attention(q, k, v)`) gives `k` a recorded consumer, but the model still holds a
  reference the wrapper can never swap. Per the spec this case has no fix: the region
  either ends before `k` or leaves `k`'s production to the library. Writing `k` out
  would not help, because the cache's reference points at the library lazy that no
  substitution map can reach. Both checks run; disagreement resolves toward retained.
- `mx.compile` wrapping: a model that compiles part of itself gets that part recorded as
  one opaque call (op = `compiled_fn`, inputs and outputs recorded). Opaque calls can
  never be inside a region, but they anchor completeness.
- The tracer installs before the model file is imported, because a model file that did
  `from mlx.core import matmul` at import time would keep the unwrapped function.
  The CLI enforces the ordering; the tracer refuses to install if the model module is
  already in `sys.modules`.

### 5.2 The two intercept modes

Record mode (fat, never timed) and bind mode (slim, always present in a patched model)
are different installations of the same patch points. When both are needed (retrace of
a patched model), record mode installs OUTSIDE bind mode: the recorder sees both the
member calls and the custom dispatch the wrapper fires.

- Record mode captures everything and is Python overhead on every op. No timed number is
  ever taken with it installed.
- Bind mode is what ships. It wraps the same global op surface but does almost nothing
  per call. Its state is an instance-keyed registry built at install time: addresses
  from the bind table are resolved against the concrete model object, so the wrapper
  fires only for the registered instances. A second, structurally identical model in
  the same process passes through untouched; its calls hit only the cheap
  not-registered fast path. (That isolation is what lets two arms coexist for retrace
  and debugging. The TIMED baseline legs of e2e go further and run with the patch set
  uninstalled entirely, per the overhead-charging rule below.)
- Per registered copy of a region, the wrapper: captures input references at the member
  calls named by the copy's `input_map` (letting the member call proceed into the
  library, returning a lazy array); at the arm address, calls the custom kernel with the
  captured inputs and enters (member_output -> kernel_output) pairs into the
  substitution map; on every subsequent intercepted call, translates any argument found
  in the map. Earlier members' library lazies are never evaluated once every consumer
  reads the kernel's outputs instead, and MLX's laziness means never-evaluated is
  never-computed.
- The top-level wrapper delimits the step: it resets call counters on entry and, on
  exit, translates the returned tree through the substitution map before clearing it.
  Without that last translation, a region output that is also a step output would be
  handed to the caller as the dead library lazy and the whole region would silently
  compute through the library anyway. A fixture pins this case (Section 12).
- Bind-mode overhead is charged to the candidate by construction: e2e times the patched
  model with bind mode installed, and the artifact installs the same wrapper. For that
  charge to be real, the baseline legs of the interleaved e2e run with the global patch
  set UNINSTALLED (cheap setattr swaps around each leg; pairing is preserved),
  otherwise the global-hook tax is common-mode across both arms and cancels out of the
  veto, letting an overhead-eaten win pass. If overhead eats a win, the e2e veto then
  rejects the ship, which is the correct outcome.
  Spike the global-hook tax in Milestone 0 so you know it early. If it ever proves
  intolerable, narrowing must keep consumer coverage total (dunders and module calls
  stay wrapped; only unused free-function wrappers may drop), because any unwrapped
  consumer of a substituted intermediate silently evaluates the dead library lazy: the
  values stay right and only the win quietly evaporates.

### 5.3 Address resolution

Addresses are strings for serialization; execution always dispatches on object
identity. `apply()` and harness bind both walk the fresh model, resolve every
`module_address` in the bind table to a live module instance (erroring loudly on a miss,
since an earlier shipped change or a model edit can restructure the tree), and register
those instances with the wrapper. This is also what makes the interleaved e2e sound:
identical address strings on two model instances never collide because registration is
per instance.

### 5.4 Trace replay

`trace/replay.py` is the one module that re-executes a recorded node sequence, and
three consumers call it: region-clock pricing (run the region's ops on saved inputs),
the fp32 golden (same ops, promoted dtypes), and sweep reference generation (library
outputs at swept sizes). Contract: the op-string-to-callable resolver is a standalone
table built by importing `mx` and resolving module attribute paths and `mx.array`
dunder/method names (a slice read replays as `mx.array.__getitem__`); the patch
installer consumes this same table rather than defining it, so replay works in a
patch-free subprocess where no tracer was ever installed (which is where gates 5, 8,
and 9 use it). Replay folds over nodes in `seq` order,
binding `array_id`s to live arrays (starting from provided bindings for region inputs
and weights), invoking the saved originals with `scalar_args` verbatim, and returns the
arrays for requested output ids. Hooks: a dtype-promotion transform and an op
substitution table (for the golden, Section 7.8). Replay never goes through the patch
surface, so it works identically whether or not any tracer is installed.

### 5.5 Region fingerprints (copy grouping)

Two stretches are copies of one region iff their canonical forms match: the sequence of
(op name, dtypes, ranks, non-shape scalar_args) with array identities replaced by role
indices (input #0, intermediate #2, weight #1, ...), including the internal edge
structure. Concrete shapes and shape-derived scalar args (reshape targets, split
sizes) are deliberately NOT part of the canonical form; they are recorded per
instance. That implements both halves of the spec's identity rule: 32 copies of a
per-layer stretch are one region, AND the same sequence firing in two workloads at two
shapes (the same norm in midbatch and decode) is still one region, priced by its
combined cost, with one hypothesis history and one wrapper, whose first attempt must
be one kernel correct everywhere and faster everywhere. Weights count by dtype and
role, not by value. The fingerprint is a hash of the canonical form. Cross-sweep-size
matching (Section 5.6) keys on module addresses instead, because copies at different
addresses are exactly what the fingerprint must unify.

### 5.6 Named dims and the sweep

The trace records concrete integers; the manifest's named dims exist only in workload
input shapes. The bridge is retracing, not symbolic shape inference: to instantiate a
region at sweep size L=13, run pass 1 again at that size (lazy, no GPU work, cheap) and
locate the region's node span in the new trace by matching (op sequence,
module_address, position_in_module), ignoring shapes. Addresses are stable across sizes
for the same model, which is what makes this sound. The matched span yields the correct
shapes AND the correct shape-derived `scalar_args` (reshape targets, split sizes) at
that size for free, because the model itself computed them. Store the result on the
region as `sweep_instances`. A region whose sequence does not occur at some sweep size
(a branch went the other way) has nothing to check at that size beyond the wrapper's
fallback engaging, and the log says so. Boundary inputs and library references at swept
sizes come from a boundary-capture pass (Section 5.7) run at that size, on demand, the
first time a region reaches gate 7.

### 5.7 Boundary capture

Pricing and every later gate run against saved region-edge tensors. Mechanics:

- Capture is a recorded pass with retention: re-run the model on the workload tensors
  with the recorder armed, hold the arrays at candidate boundaries, `mx.eval` them, and
  write to safetensors. Matching the pass's calls to the priced trace's nodes is by
  call sequence index, under the determinism assumption fixed inputs and seeds give;
  each node is verified against (op, shapes, module_address) and a mismatch aborts
  naming the divergent seq.
- Capture runs in batches under a resident-bytes budget: candidates overlap heavily, so
  retaining every boundary in one eval approaches every intermediate in the model and
  will not fit in memory at real scale. Multiple passes per workload, each retaining a
  subset, flushing to disk and freeing between passes; extra passes are off-clock and
  free. Weights appearing as region inputs are captured once and referenced by id
  across all stores.
- k distinct input sets per workload (default k=3, recorded) are produced by
  materializing the workload k times under k recorded seeds and running the capture per
  seed. The rotation that keeps region clocks honest (Section 6, law 7) and the value
  variation the anti-caching checks need (Section 7) both come from these k sets;
  nothing else manufactures input diversity for correctness.
- k=3 is a floor, not the criterion. The spec's requirement is functional: enough
  distinct sets that the rotated data cannot just sit in the GPU's cache. So the
  region clock checks the rotated boundary bytes against a cache-defeat threshold
  (default 128MB, tunable) and, for small regions, synthesizes extra timing-only input
  sets at matching shapes and dtypes until the threshold clears. Correctness
  comparisons always use the real captured sets only; synthesized sets exist purely to
  keep memory traffic honest in the clock.

### 5.8 Launch grammar

`grid`, `threadgroup`, and `fallback_predicate` are expressions over the region's
shape environment: named dims plus `in0.shape[i]`-style accessors, with integer
arithmetic (`+ - * / // % min max ceil_div`) and comparisons/boolean ops for
predicates. The harness parses them (a ~50-line expression evaluator; never `eval`),
static checks validate them, and the sandbox and wrapper evaluate them against actual
shapes at every call. The grammar is documented in the judge prompt (Section 10). This
is what lets one kernel launch correctly at every sweep size, and what makes a
declared-but-dead fallback detectable.

### 5.9 Scaffold strategy

Naive lowering is the required, always-available path: emit one Metal kernel that runs
the region's stages back to back, intermediates in plain device memory, correctness
first, no cleverness. It only needs to pass the ladder, not to be fast; the spec is
explicit that a slow scaffold is normal. Build the lowering as a small library of
per-op emitters (elementwise, reductions, matmul, the `mx.fast.*` primitives the trace
can contain, `mx.quantized_matmul`) plus a shape/stride indexing helper. Emitters must
handle the region's recorded ranks generically and take sizes from the launch-grammar
environment, because the sweep moves named dims.

The spec's priority order stands: stitch from the library's own kernel source where
possible, naive otherwise. "Possible" is defined here as: the include-flattened
shipped source covers the region's dominant op and the stitched result passes the
ladder. Naive lowering is built first because it must exist for every region either
way. The wheel ships ~4MB of MSL under
`site-packages/mlx/include/mlx/backend/metal/kernels/` (steel GEMM, sdpa, quantized,
reductions, all plain text). Two facts (re-verify): `mx.fast.metal_kernel(header=...)`
resolves absolute-path `#include` lines from disk, but the shipped headers' own nested
includes are repo-relative and do not resolve, so stitching requires a one-time include
flattener that inlines each entry point into a self-contained header. Build the
flattener; stitch wherever the definition above holds; fall back to naive lowering
everywhere else.

### 5.10 Kernel compilation and evaluation facts (all re-verify in Milestone 0)

- `mx.fast.metal_kernel(name, input_names, output_names, source, header='',
  ensure_row_contiguous=True, atomic_outputs=False, compile_options=None)`. `source` is
  the kernel BODY; MLX generates the signature. The call is keyword-only:
  `kernel(inputs=[...], output_shapes=[...], output_dtypes=[...], grid=(x,y,z),
  threadgroup=(x,y,z), template=[('T', dtype), ...], init_value=None, verbose=False)`.
  `grid` is TOTAL THREADS (dispatchThreads semantics), not threadgroup counts.
- The harness owns every kernel call site. The judge supplies only source, header,
  template names, and the launch expressions; `init_value`, `math_mode`, streams, and
  everything else are harness-set. This is enforcement by construction, not by scanning.
- `init_value=nan` fills output arrays before launch (verified on 0.32.2: unwritten
  elements read back NaN deterministically). This is the primary output poison during
  correctness gates. It is never set on timed runs (the fill pass would tax only the
  candidate).
- Referencing `inp_shape` / `inp_strides` / `inp_ndim` in the source auto-injects those
  buffers; grid built-ins appear in the generated signature only if the source text
  mentions them. A helper header (`utils.h`, ~471 lines) is auto-prepended.
- Compile errors surface as `RuntimeError` at `mx.eval` time, never at construction or
  call. The harness must eval a probe output under try/except to detect a broken build,
  and must subtract the auto-header line count when reporting error line numbers to the
  judge.
- Fast-math is a per-kernel knob: `compile_options={'math_mode': 'safe'|'relaxed'|'fast'}`.
  Pin `safe` for every kernel in the job, harness-side. Record the pin in the report.
  (`fast` silently breaks NaN and inf semantics.)
- `ensure_row_contiguous=False` with naive flat indexing is silently wrong on strided
  views. Kernels default to `True`; a kernel that opts out must index through the
  injected strides, and the sweep's transposed inputs exist to catch the ones that lie.
- Metal does not reliably fault on out-of-bounds access; observed OOB reads returned
  zeros silently. Correctness is value comparison only, never "it didn't crash".
- The allocator recycles buffers without zeroing, and a fresh process gets OS zero
  pages. `init_value` closes the output-buffer hole; pool saturation (allocate
  NaN-filled buffers of the reference computation's output sizes, eval, free, repeat)
  is kept as belt and braces around library reference runs.
- Shader validation (`MTL_SHADER_VALIDATION=1`) and GPU capture
  (`MTL_CAPTURE_ENABLED=1`) are read by Metal at process launch; setting them inside a
  running process does nothing. This forces the subprocess architecture in Section 8.
- Float atomics are nondeterministic across runs (spike it; the determinism gate exists
  for this).
- Kernel caching: same-name-different-source kernels behaved independently in one
  process on 0.32.2, but the on-disk shader cache across processes was never tested. A
  stale-binary hazard would poison every verdict; spike it in M0.

### 5.11 The judge client

- Anthropic API, model configurable per job (default to a current top-tier model),
  temperature low. All judge I/O is strict JSON validated against `judge/schema.py`;
  a malformed response gets one re-ask, then counts as a failed hypothesis. It
  consumes hypothesis budget (a babbling judge must exhaust its region, not stall it)
  but never counts toward the 5-straight compile/static fail streak, which is about
  kernels on one parent.
- The judge is stateless per call. Every call re-renders the region's state: shapes,
  dtypes, p, bound, head and shipped distance from roofline, the assoc tag of the
  kernel being edited (the spec's `family: assoc-preserving | assoc-changing` line),
  the parent kernel source, the last verdict with its gate detail, the queue, and the
  menu. No conversation
  history accumulates; the queue and the verdict log ARE the memory. This keeps context
  small and makes every judge call reproducible from the run log.
- Two judge entry points only: `seed(region_meta) -> queue` and
  `next(region_meta, verdict) -> {queue mutations, kernel for front ready item}`.
  Plus the one scaffold-fix attempt, which is `next` with the scaffold failure as the
  verdict.
- `judge/scripted.py` implements the same interface from a canned script. All loop and
  ladder tests run against it; no test depends on a live LLM.

### 5.12 Constants

Two kinds. Spec-fixed values may never be changed by config. Plan defaults concretize
the spec's loose words ("about", "a few", "tens of"); they are manifest/config tunable
and every used value is recorded in the report.

Spec-fixed:

| Constant | Value |
|---|---|
| Watchdog | 10x library region time |
| Ship margin | max(1% of the library region time re-measured in this verdict, 3 sigma of the interleaved samples) |
| Fail-streak close | 5 straight compile/static fails on one parent |
| Roofline close | shipped within 5% of roofline |
| Diminishing-ships close | 3 ships in a row, each under 2% better than the last |
| Head-near-roofline close | head minus roofline under ~1% of the step |
| Family abandonment | 8 correct-but-slower without ever beating the library |
| Scaffold fix attempts | 1 |
| Determinism runs at gate 8 | 3 |
| Stale-hypotheses close | 6 in a row beating shipped by neither 1% of region time nor the minimum absolute win (the "tens of microseconds" term uses the tunable default below) |

Plan defaults (tunable, recorded):

| Constant | Default | Spec language |
|---|---|---|
| Region floor | 2% of step (copies combined) | "about 2%" |
| Roofline has_room | s_max >= 1.2 | "barely beats ... is skipped" |
| Launch cost per kernel (T_launch) | measured at job start via empty-kernel chain; fallback 4 us | "a few us" |
| Minimum absolute win | 30 us per step across copies | "a few tens of microseconds" |
| Assoc-changing kappa | 1.25 | "around 1 to 1.5" |
| Assoc-changing floor | spread of library-vs-golden error across the k input sets, floored by a small multiple of output-dtype epsilon at observed scale | "+ floor" |
| Sweep defaults per named dim | 1, 13, 50, 4096 | "1, a prime, a non-multiple of 32, one large" |
| budget.per_region | 25 hypotheses | defaulted, recorded |
| budget.total | 250 hypotheses | defaulted, recorded |
| e2e step veto | patched not slower than max(0.5%, 3 sigma) under interleaved pairing | "not significantly slower" |
| Boundary input sets | k=3 per workload (a floor; timing rotation grows past it per the cache-defeat rule) | (Section 5.7) |
| Cache-defeat threshold | 128MB rotated working set for the region clock | "enough that the data cannot just sit in the GPU's cache" |
| Warm-until-stable | two consecutive timings within 1%, cap ~30, every distinct shape warmed once | spec fixes only "first eval thrown away"; this strengthens it |
| Tolerances (assoc-preserving) | fp32 rtol 1e-5 atol 1e-6; fp16 rtol 1e-2 atol 2e-2; bf16 rtol 2e-2 atol 4e-2 | manifest-overridable |

(The naive floor for the assoc-changing gate cannot be "library run-to-run wobble"
alone: a deterministic library binary has zero wobble on fixed inputs and the gate
would degenerate to bare kappa. Hence the k-set spread with an epsilon floor.)

### 5.13 Sign and logging conventions

Internal math: positive delta means candidate faster. Human-facing log lines: signed
milliseconds where negative means faster, rendered by one shared formatter. The
convention statement is embedded in `report.json` itself. The run log is append-only
JSONL with elapsed seconds on every row; every verdict row carries the gate detail, and
every close carries which rule fired.

---

## 6. Measurement laws

These are load-bearing. Each one exists because the naive version produced a measured,
named failure on this class of hardware. Bake them into `measure/` as invariants, not
conventions. Numbers marked (motivating history) are illustrative; numbers the build
relies on have M0 spikes.

1. **Thermal drift is the dominant confound.** Sustained GPU load inflates latency
   40-55% and the noise floor 4-9x, onset after roughly 60-120 seconds of continuous
   work (motivating history; the M0 duty spike re-establishes the shape on the actual
   machine). Heat inflates variance, not just the mean, so it cannot be averaged or
   normalized away.
2. **Duty-cycle pacing.** After every measured chunk of w seconds of GPU work, idle ~3w.
   Enforce in `measure/session.py` as an invariant around every timed path (clocks,
   peaks, e2e). Validate on new hardware with a blocked (not interleaved) long-run A/A
   test: interleaving duty levels measures nothing because heat is accumulated state
   (M0 spike).
3. **Warm until stable, not a fixed count.** Warm every distinct shape once, then repeat
   until two consecutive timings agree within 1% (cap ~30). Fixed small warmup counts
   have left first-ever kernels reading 2x slow (pipeline JIT plus page faults;
   M0 spike pins convergence on a first-ever kernel). The spec's rule that the first
   eval of each workload is thrown away is the floor, not the ceiling.
4. **Pair and interleave everything comparative.** A/B comparisons run in one session,
   alternating (the spec's ABBA for the ship clock), so drift is common-mode. Never
   subtract two separately timed quantities; that fabricated a large phantom effect
   once (motivating history). Baselines are re-measured at every verdict and never
   reused.
5. **Medians for comparisons, running max for peaks.** Peaks are microbenched once at
   job start, per the spec, but the estimator inside that calibration is the running
   max over samples spread across the window, because throttling, contention, and cold
   start all push observations DOWN only. Mid-job recalibration is optional, monotone
   (may only raise a peak), and logged; it deliberately refines the spec's
   measure-once wording, because a cold-start under-measurement would otherwise
   understate every roofline for the whole job. Flops peaks are measured per dtype
   present in the trace (fp16, bf16, fp32 differ on some generations); a region's
   roofline uses its own compute dtype's peak. Comparisons use medians with spread.
   Two quantities, two estimators, never swap them.
6. **Timing recipe.** `mx.eval` the callable's outputs, bracketed by `mx.synchronize()`,
   `time.perf_counter` around it. MLX has no timing helpers. Beware
   `MLX_MAX_OPS_PER_BUFFER` and friends in the environment: they change command-buffer
   batching and shift wall times, so the harness records the relevant env vars in the
   report and refuses to compare runs across different values.
7. **Defeat laziness in every timed loop.** MLX computes only what is observed. Every
   region-clock iteration's outputs go into the final eval (the spec's rule), inputs
   rotate through the k captured sets so the cache cannot make memory traffic free, and
   any value the timed computation is supposed to read must actually influence an
   evaluated output.
8. **No absolute-time vetoes.** Absolute times on a busy Mac swing 3x while paired
   deltas stay consistent; absolute-time health gates have blocked real wins
   (motivating history). Machine health is observed (log baseline drift per verdict;
   baseline latency tracks the noise floor monotonically and is the only free
   thermometer), never enforced. Protection lives in the paired design and the 3-sigma
   term of the ship margin.
9. **Memory pressure is noise.** Holding a few extra GB resident has tripled the A/A
   floor (M0 spike re-checks). Patched and baseline arms share weight arrays; boundary
   stores stream from disk rather than living resident; capture runs batched under a
   resident budget (Section 5.7); the report records peak memory. The two e2e arms are
   two `build()` results with the second model's parameter arrays replaced by the
   first's (verify sharing by array identity). If a model's `build()` cannot share
   weights, both arms stay resident anyway and the report notes the memory pressure:
   the spec requires the step veto to run under the interleaved discipline, so the
   extra noise hardens the veto's 3-sigma term instead of the comparison going
   unpaired.
10. **Fresh callables after any swap.** `mx.compile` caches on callable identity and
    silently returns the pre-swap graph (every candidate then measures exactly 0.00%).
    The harness never compiles, but a model whose own forward compiles internally must
    get a freshly built callable after every bind before any timed pass. (M0 spike
    pins the caching behavior.)
11. **The measurement floor is real.** On the reference machine, ~1.5% end-to-end was
    reliably detectable and ~1% was a coin flip (M0 A/A spike re-establishes the
    floor). This is why the ship clock is the region clock (a 2x win on a 3% region is
    invisible at step scale) and why the e2e step check is a non-regression veto, not
    the detection gate.
12. **Controls are part of the harness, not the tests.** A/A nulls (same kernel both
    arms) must not ship; a planted, bit-identical slowdown of known size must be
    detected. Ship both as executable controls runnable per session, and run the A/A
    null automatically at job start to calibrate and log the session's floor.

---

## 7. The ladder, bind, and e2e

The spec fixes the gates and their order. Implementation notes per gate, then the two
promotion checks.

1. **Static checks** (no GPU): input and output names, counts, and ranks match the
   region's; boundary dtypes identical to the region record; no dtype change in launch
   config or output dtypes; no dequantize-and-store of weights in a new format; every
   live value still produced or listed as an output; shape-specialized kernels declare
   a fallback predicate in the launch grammar, and the predicate parses. The law "no
   new cast or quantization" is scoped to the model-visible boundary on purpose:
   internal accumulator precision (an fp32 accumulator in an fp16 kernel) is legal
   kernel-schedule territory, which the library itself uses, and the numeric gates
   police its effects. Tolerance secrecy is structural, not textual: the prompt
   renderer has no access to tolerance values and the sandbox compares against
   harness-held constants, so there is nothing to scan source for.
2. **Compile**: build the pipeline in the sandbox via a probe eval on minimal inputs;
   catch `RuntimeError`, strip the Metal preamble, fix line numbers (auto-header
   offset), return structured diagnostics.
3. **Poison**: every correctness-gate launch of the candidate runs with
   `init_value=nan`, so an unwritten output element is NaN deterministically, not
   probabilistically. Pool saturation additionally runs before library reference
   computations. Re-poison between the determinism runs.
4. **Watchdog** on the first workload. Two mechanisms, not one: the parent enforces a
   generous absolute wall timeout on the whole subprocess (kill-and-relaunch; process
   death is how a wedged GPU recovers on this platform, treat it as routine). Inside
   the child, the 10x rule compares the kernel's first timed run against a library
   region time re-measured in the same process and mode, so validated compares against
   validated.
5. **Smoke numerics** on the first workload: every output compared under the gate-8
   numeric rule against the saved library reference; finite unless the reference is
   non-finite there. Then two extensions the spec's wording permits and the cheat zoo
   requires: (a) value variation, the same comparison across the k captured input sets,
   which kills cached-by-shape kernels; (b) adversarial value regimes at the recorded
   shapes (inputs scaled up 1e3, scaled down 1e-4, outlier-injected, zeros, planted
   inf/NaN lanes), with the library reference computed on the spot in the sandbox via
   replay, never stored anywhere the judge's code path can reach. Plain randn is too
   well-behaved to expose sloppy accumulation; the regimes are what catch it. The
   on-the-spot references are a narrow, intentional exception to the spec's rule that
   later checks run against the saved tensors: the replay is region-sized, so the
   rule's purpose (never re-pay the rest of the model per region) still holds.
6. **All workloads** the region fires in: same as 5.
7. **Shape sweep**, correctness only, never speed. Named dims move through the sweep
   sizes; integer dims never move. Region instantiation at each size comes from
   `sweep_instances` (Section 5.6); inputs and library references come from an
   on-demand boundary capture at that size. Include at least one non-contiguous
   (transposed) input variant: it is the documented killer of stride-lying kernels.
   For shape-specialized kernels, the sandbox itself evaluates the fallback predicate
   at each sweep shape, routes to the library when it fires, and logs the engagement;
   a fallback that never engages anywhere it should is a fail (the
   declared-but-dead cheat). The real wrapper's fallback is re-verified at the job's
   final sweep pass (M10).
8. **Determinism, then numerics**: three runs bitwise identical (both assoc tags; a
   racy kernel dies here regardless). Assoc-preserving: match the library within
   manifest tolerances, floored by the library's own run-to-run wobble measured on the
   spot. Assoc-changing: build the fp32 golden by replaying the region's recorded ops
   (Section 5.4) with activations promoted to fp32 and an op substitution table for
   quantized ops (`quantized_matmul` -> `dequantize` + `matmul` in fp32, pinning
   group_size/bits from `scalar_args`; extend the table as traced ops require), then
   require err(candidate) <= kappa * err(library) + floor, where err is the worst
   per-output max relative error against the golden (denominator clamped) and the
   floor is Section 5.12's. The tag is a claim: an attempt tagged preserving that
   fails the tight gate may be resubmitted as changing.
9. **The region ship clock**: looped replay on the saved boundary inputs, candidate and
   library interleaved ABBA in one session, tens of milliseconds of work per sample,
   the k input sets rotating, all outputs kept live, baseline re-measured now.
   Ship iff median win > max(1% of that freshly re-measured library region time,
   3 sigma of the interleaved samples) AND win times copies >= the absolute floor.
   (The spec says "1% of region time" right after saying the baseline is re-measured
   now and never reused; the fresh measurement is that region time. Pricing's T_orig
   is stale by construction here.) The full model is not run here; that is bind's job.

Gates 1-8 run with shader validation ON (validation subprocess); gate 9 runs in a clean
subprocess with validation OFF, and re-runs smoke and determinism first, because
validation recompiles instrumented pipelines and the timed pipeline must be the checked
pipeline (spec rule; the two-subprocess split is forced by Metal reading the env var at
launch).

### 7.10 Bind verification (retrace)

A literal "the op stream now shows one call" check is impossible by the bind mechanics
themselves: member ops still execute at Python level and still get recorded; only their
outputs go unused. So `bind/verify.py` is a dataflow check on the frozen retrace
(record mode installed outside bind mode):

- the custom-kernel node is present at each copy's arm address, consumes the cut's
  inputs, and its outputs reach every downstream consumer and live value the region's
  outputs previously fed;
- the member nodes' library outputs are unreachable from step outputs and retained
  values (so laziness never evaluates them);
- after deleting those dead member nodes, the remaining node sequence matches the
  baseline trace under (op, specs, module_address) alignment: neighbors unchanged.

Any miss is a failed bind: roll back, tell the judge.

### 7.11 E2e

- **Floor**: run the original model twice on the workloads; the orig-vs-orig
  differences (max abs, plus cosine similarity on the outputs, or KL where outputs are
  logits) are the floor.
- **Assoc-preserving ships**: patched vs original must sit on that floor. A large miss
  means the bind installed the wrong cut.
- **Assoc-changing ships**: the golden-relative rule at model scale, per the spec:
  weights stay quantized, activations run in fp32. Mechanism: reuse the patch surface
  as a dtype-promoting interceptor that casts floating array-producing op inputs to
  fp32 and routes quantized ops through the golden substitution table, and run the
  ORIGINAL model under it, once, to produce the model-scale fp32 golden. Both arms are
  then compared in their normal dtypes against that golden: require
  err(patched) <= kappa * err(original) + floor at the model outputs, with the floor
  being the spread of original-vs-golden error across the k input sets, epsilon-floored
  (the region-scale floor in Section 5.12, lifted to model scale). The patched model is
  never run under promotion; its kernels are compiled for the frozen dtypes and would
  either reject fp32 inputs or fall back to the library, making the check vacuous.
  This is real machinery; it is scheduled in M8, and the M12 flagship (quantized
  decode) will exercise it, because any win there is almost certainly assoc-changing.
- **Step-time veto**: patched vs original under the same interleaved paired discipline;
  patched must not be slower than max(0.5%, 3 sigma). A veto, not a detection gate.
- Both arms live in one process, weights shared (law 9); the wrapper's instance keying
  (Section 5.3) keeps the baseline arm clean, and M8's acceptance includes proving the
  baseline arm fires zero custom dispatches.

Failing bind or e2e rolls the ship back and the judge mutates the queue.

---

## 8. Subprocess execution model

Every kernel evaluation is out of process. One `sandbox/worker.py`, three launch modes,
selected by environment at spawn:

- **validate mode**: `MTL_SHADER_VALIDATION=1`. Runs gates 1-8.
- **score mode**: clean env. Re-runs smoke + determinism, then gate 9. Also used for
  the step clock, region pricing, bind retrace, and e2e once M5 lands (M3/M4 may
  measure in-process with an asserted-clean environment; M5 migrates them and checks
  the numbers agree).
- **capture mode** (debug only, off the hot path): `MTL_CAPTURE_ENABLED=1`, wraps a
  single dispatch in `mx.metal.start_capture` for humans to replay in Xcode.

Protocol: parent writes one JSON job spec (manifest ref, region fingerprint, boundary
store paths, kernel source + launch expressions, gates to run, seed) to the child's
stdin; child prints exactly one JSON verdict line to stdout as its last line; parent
enforces the wall timeout and maps nonzero exit or timeout to a structured
`{failed_gate: 'subprocess', detail: stderr tail}`. The child rebuilds everything from
the spec (imports the model file, loads saved tensors); nothing is pickled, no state is
shared. This is also what guarantees every shipped kernel is reconstructible from its
serialized form alone.

---

## 9. Milestones

Each milestone ends with tests green and an acceptance demo runnable from the CLI.
Build in this order; it puts a working measurement kit under everything, and the
existential risks (tracer completeness, bind mechanics) are spiked in M0 before
anything depends on them.

### M0: Bootstrap and platform spikes

Repo scaffolding, packaging, pytest, `PLATFORM.md`, `UNVERIFIED.md`. Then a spike
script per platform fact this plan relies on, each producing a pass/fail line and,
where possible, graduating into a permanent pinned test ("tests that encode findings"):

- Patchability: module-level `setattr` seen by `mlx.nn` layers; array dunder patching
  propagates to slots; the full dunder list including reflected and in-place ops;
  `__eq__`/`__hash__` patching does not break dict/set use of arrays; `a += b` folding.
- Module wrapping: `nn.Module` has no `__call__` of its own; per-subclass
  `type(m).__call__` patching intercepts a Linear call (verified once on 0.32.2,
  pin it).
- Weakref/GC probe on `mx.array` (verified once on 0.32.2, pin it), and the positive
  retention snapshot walk on a toy stateful model.
- Bind mechanics micro-spike: on a three-op toy chain, hand-wrap the op surface, let
  the first two ops return library lazies, fire a hand-written fused kernel at the
  third, substitute its outputs into the consumer, and verify the library lazies are
  never evaluated (laziness check: wrap them so evaluation would raise) while outputs
  match. This is the whole bind idea in fifty lines; if it does not work, stop and
  redesign before M1.
- `metal_kernel`: construction/call signatures on the pinned version; grid semantics;
  probe-eval error surfacing and the line offset; `compile_options.math_mode`;
  `init_value=nan` poisons unwritten outputs (verified once on 0.32.2, pin it);
  same-name-different-source cache behavior in one process AND across processes (the
  on-disk shader cache was never tested; a stale-binary hazard would poison every
  verdict); float-atomic nondeterminism.
- OOB read behavior with and without validation (expect silence; confirm the design
  cannot depend on faults).
- `MTL_SHADER_VALIDATION` and `MTL_CAPTURE_ENABLED` are launch-time only.
- Include flattener feasibility: absolute-path includes resolve; nested repo-relative
  includes fail; a flattened steel GEMM header compiles.
- `mx.compile`: identity caching, state freezing, per-shape retrace; and
  plain-vs-compiled step time on a small test net (informs nothing, but records the
  baseline-is-plain rule's context on this machine).
- Bind-mode overhead estimate: a do-nothing global wrapper on every op surface, cost
  per op call and per decode-shaped step.
- Measurement: A/A nulls cool and after sustained load (the session floor);
  warm-until-stable convergence on a first-ever kernel; duty-cycle validation
  (blocked design); memory-pressure probe (A/A floor with extra GB resident);
  peaks microbench per dtype with the running-max estimator.

Done when: `PLATFORM.md` states each fact with its spike result on the actual machine,
and any fact that failed has a design adjustment recorded in this plan before further
building.

### M1: Measurement kit (`measure/`)

Session pacing, warm-until-stable, the timing recipe, paired interleaved comparison
with per-sample records, step clock, per-dtype peaks with running max, the A/A control,
and the bit-identical slowdown injector (inject redundant work that provably equals
existing values, e.g. recompute a reduction and average it with itself, so outputs are
bit-identical but the work is unfoldable).

Done when: the A/A control reads ~0 with the session floor logged; the injector wraps a
synthetic looped callable producing tens of milliseconds of work per sample, and its
known 2-3% slowdown is detected reliably through `clocks.py`'s paired comparison; peaks
are stable across a hot/cold cycle; every number in the acceptance run comes from the
public API of `measure/`, nothing ad hoc.

### M2: Manifest and workloads

Parsing, validation, defaults, named-dim handling, sweep sizes, seeded tensor
materialization (k seeds per workload), the subprocess-importable `build()` contract
with its error messages (a live object, a closure, or a build that reads CWD-relative
paths must fail with a message that says what to fix).

Done when: good and bad manifests round-trip through tests; the same manifest and seed
materialize byte-identical tensors in two processes.

### M3: Tracer (record mode)

The patch sets, node capture, array identity, per-subclass module wrapping, top-level
wrapping, `mx.compile` wrapping, freeze with edges + liveness + completeness, replay.
Then the step clock integration: per workload, pass 1 records lazily, pass 2 times with
the recorder fully uninstalled. In-process measurement is legal here with an asserted
clean environment; M5 migrates it.

Done when, on the fixture zoo (Section 12): operator-heavy models record completely;
a model using an unwrapped entry point aborts naming the call; reflected and in-place
operators appear as nodes; every node executed inside a module carries a non-empty
module_address, and a plain-function model's ops carry the top-level address; the
compiled-submodule fixture records one opaque call; the cache-retention fixture marks
the retained-and-consumed array `python_retained`; the eager fixture (a model calling
`.item()` mid-forward) records completely and logs the memory warning; the
data-dependent-branch fixture records only the taken path; replay re-executes a
recorded span and reproduces the library's outputs bitwise; uninstall restores `mx`
exactly (identity checks on every patched attribute).

### M4: Regions (`regions/`)

Builder (singletons, chain growth, view absorption, slice-write termination, the four
rejection rules, per-stretch liveness derivation), fingerprint grouping, batched
boundary capture with k input sets, region clock pricing, roofline, floor, ranking
with tie-breaks, and sweep-instance matching (retrace at a second size, locate every
region by address).

Done when: on fixtures, the candidate set matches hand-derived expectations (including
a norm-then-three-projections fixture merging on shared inputs, a views-only stretch
producing no region, a mid-stretch cache write splitting a region); priced shares are
stable across two runs within noise; a coverage diagnostic (sum of region clocks times
copies vs step clock) lands in a sane band and is logged; roofline bounds match
hand-computed boundary bytes and flops on fixtures, including a chain whose roofline
bytes are provably smaller than the sum of its ops' bytes; a region priced at L=512
resolves and replays at L=7 via its sweep instance; capture aborts loudly on an
injected nondeterministic model.

### M5: Sandbox (`sandbox/`)

Worker, protocol, three modes, poisoning, the two-level watchdog, kill-and-relaunch.
Ships with a minimal gate set only (compile probe, poison, watchdog, a bare
allclose-against-reference); the full ladder is M6. Migrate the step clock and pricing
paths into score mode.

Done when: a hanging kernel, a compile-error kernel, a crashing kernel, and a
partial-write kernel each produce the correct structured verdict without harming the
parent (the partial-write case must fail deterministically, via init_value); the same
job spec re-run twice gives the same verdict; in-process and score-mode step clocks
agree within noise.

### M6: Ladder (`ladder/`)

All nine gates in order, structured per-gate results, value regimes and k-set
variation, golden evaluator with the quantized-op substitution table, tolerance
compare with non-finite pattern matching, the launch-grammar evaluator, fallback
routing in the sandbox, the ship clock.

Done when: the ladder-catchable cheats in the zoo (Section 12, everything caught at
gates 1 through 9) are fully caught, each at its intended gate; the two bind-level
cheats (the live-output sabotage variant and the wrong-cut bind entry) wait for M8,
whose done-when covers them; a correct-but-slower hand-written kernel produces
`correct_slower` with honest numbers; an assoc-changing kernel (a reduction reordered
on purpose) passes only under the golden gate and not the preserving gate.

### M7: Scaffold (`scaffold/`)

Naive lowering emitters for the op set the fixtures and a small real transformer
actually record (elementwise, reductions, softmax-shaped patterns, matmul,
`mx.fast.rms_norm`/`layer_norm`/`rope`/`sdpa`, `mx.quantized_matmul`), the include
flattener, and stitched scaffolds for ops with shipped MSL where straightforward.

Done when: every fixture region gets a scaffold that passes gates 1-8, including the
sweep (launch expressions over named dims, not baked constants); a deliberately
unlowerable region reports "no scaffold" cleanly and goes straight to skip: the
judge's one fix attempt exists only for a generated scaffold that failed the ladder,
never for authoring a kernel from a blank page.

### M8: Bind, retrace, e2e (`autotuner_runtime/`, `bind/`, `e2e.py`)

The wrapper (instance registry, input capture, arm-and-substitute, step delimitation
with returned-tree translation), address resolution, bind-table format, the retrace
dataflow verification, e2e floor and both assoc paths and the veto, rollback.

This is the riskiest milestone after the M0 spike; do not let it slip. Done when: a
hand-written correct kernel for a fixture region binds; retrace verification passes and
is shown to check dataflow, not stream length; e2e sits on the orig-vs-orig floor with
step time unchanged or better; the fixture whose region output is also a step output
gets the kernel's output back from the top-level call; with both arms live in one
process, the baseline arm's retrace shows zero custom dispatches; an assoc-changing
hand kernel passes e2e only under the golden-relative path while an assoc-preserving
one sits on the floor; sabotage cases roll back cleanly (kernel writing a wrong live
output fails e2e; a bind entry pointing at the wrong position fails retrace); the
runtime package alone (no `autotuner` import) applies a bind table to a fresh `build()`
model, resolves addresses to instances, and reproduces the patched outputs.

### M9: Judge (`judge/`)

Schema, prompts, scripted judge, real client, the queue mechanics (pop-ready,
depends_on conditions, mutation after verdicts, per-family_id bookkeeping: the 8-strike
counter, head reset to scaffold or last shipped on abandonment, family deletion on ship
as a judge-side queue mutation with harness-side counters).

Done when: the full hypothesis cycle runs against the scripted judge deterministically
(seed queue, execute front item, verdict, mutate); prompt rendering is snapshot-tested;
a malformed LLM response burns one re-ask then one hypothesis, never the run; family
abandonment triggers on the 8th correct-but-slower and head resets correctly.

### M10: The region loop (`loop.py`)

Open (scaffold, fix-or-skip), the hypothesis cycle wired to sandbox + ladder + bind,
close rules, family bookkeeping, retrace-after-close with share updates and covered-
region dropping, budgets, the final e2e plus the sweep-fallback pass through the real
wrapper, full run logging.

Done when: with the scripted judge on the planted-win fixture, the whole job runs
end to end: finds the region, ships a scripted winning kernel through bind + e2e,
closes by rule, retraces, and the run log replays the spec's example history shape
(fail, fix, climb, ship, plateau, close). A second scripted run on the vendor-parity
fixture ships nothing and says so.

### M11: Artifact (`artifact/`)

Emit kernels + bind_table + vendored runtime + report; `apply()` in a fresh process.

Done when: a fresh Python process with only `artifact/` and the model file loads the
model, applies, matches outputs, and reproduces the step-time improvement within noise;
`report.json` carries the spec's full field list: per region p, s = T_orig/T_shipped,
and step before/after; per hypothesis id, kind, parent, verdict, failed gate, and
region ms; plus all pinned seeds, versions, env vars, and the defaults actually used.

### M12: Real-model validation

Three runs, in order:

1. **Positive control**: a small transformer-ish model with a deliberately unfused,
   hand-rolled chain (the planted-win fixture scaled up), real judge. The loop must
   ship something real.
2. **Negative control**: a stock small model whose ops are already vendor-optimal at
   the declared workload. The loop must ship nothing, and the final e2e must read ~0%.
3. **Flagship**: a 4-bit quantized decode-shaped workload at small batch (M in 2..24).
   Prior audit measured `mx.quantized_matmul` there at 3-5x over its roofline floor
   (worst near M=4), the speculative-decoding and parallel-sampling regime, while M=1
   and M>=32 were near the wall. Those numbers have no surviving logs: re-run the
   bare-kernel M-sweep first (distinct pre-evaluated inputs, batched evals, medians,
   floor = max(weight-stream bytes / measured bandwidth wall, flops / measured
   flops wall)), then point the tool at it. This is the first honest test of the whole
   thesis: a real vendor gap, reachable by a specialized kernel, scored end to end,
   and almost certainly exercising the assoc-changing e2e path.

Done when: all three runs' reports are archived, and the flagship either ships a real
win or the report shows precisely which gate real candidates die at (either is a
result; only silence is failure).

---

## 10. What the judge sees (prompt contract)

Rendered per call, JSON, no prose padding:

```
region:
  shapes/dtypes of inputs and outputs, per workload
  p, copies, bound (memory|compute|launch), T_orig per workload
  s_max and the distance of head and shipped from T_roofline
family: assoc tag of the kernel being edited (assoc-preserving | assoc-changing)
families: per family_id, climb state and strike count; current family being climbed
parent: kernel source + launch expressions being edited, and its verdict
last_verdict: outcome, failed gate, gate detail (compiler diagnostics with fixed
  line numbers, worst numeric excess, timing samples)
queue: current items with satisfied/unsatisfied conditions
menu: on-chip intermediates, specialize launch/tiles, retile, re-layout
  (must round-trip exactly), same-math different algorithm, launch changes, fix
launch_grammar: the expression language for grid/threadgroup/fallback_predicate
laws: boundary dtypes frozen, no new quantization, no approximate math, fallback
  required for shape-specialized kernels, fast-math pinned by harness
```

The judge returns queue mutations plus, for the front ready item, one Metal body,
optional header, launch expressions, and template names, as an edit of a named parent.
Nothing else crosses the boundary in either direction. Enforcement is structural: the
prompt renderer has no access to tensors or tolerance values, and the harness owns the
kernel call site (Section 5.10), so `init_value`, `math_mode`, and streams are simply
not the judge's to set.

---

## 11. Coverage diagnostics

Two standing self-proofs, logged every job:

- After pricing, the coverage line: sum of region clocks times copies vs the step
  clock. Overshoot of a few percent is launch overlap and fine; a large shortfall
  names the fraction of the step running outside every candidate.
- After every close, the retrace updates every remaining region's p and drops regions
  covered by a shipped larger cut, per the spec; the log records both.

---

## 12. Test strategy

- **Fixture zoo** (`tests/fixtures/`): tiny models, each existing to exercise one
  mechanism. Operator soup (dunders, reflected, in-place, slicing, slice-write);
  norm + three projections sharing an input (chain growth without data edges);
  views-only stretch; mid-stretch Python retention (a cache object, retained AND
  consumed); a region output that is also a step output; a plain-function model (no
  nn.Module anywhere); an eager model calling `.item()` mid-forward; data-dependent
  branch; compiled submodule; repeated identical layers (copy grouping); a planted-win
  model whose chain a naive fusion beats; a vendor-parity model. Fixtures are also the
  M12 controls at larger scale.
- **Cheat zoo** (`tests/cheats/`): the harness's job is rejecting bad kernels, so its
  tests ARE bad kernels, each asserting WHICH gate catches it: partial-write (gate 3
  poison via init_value); shape-hardcoded (gate 7 sweep); stride-lying (gate 7
  transposed input); cached-by-shape (gate 5 k-set value variation);
  cached-by-shape-and-magnitude (gate 5 value regimes, which change magnitudes);
  fp16-sloppy accumulation (gate 5 regimes: large-scale and outlier inputs);
  eps-dropping (gate 5 tiny-scale regime); atomic-racy (gate 8 determinism);
  live-output-dropping (gate 1 static, and e2e for the sabotage variant);
  fallback-missing (gate 1); fallback-declared-but-dead (gate 7 fallback log);
  wrong-cut bind entry (retrace verification).
- **Tests that encode findings**: every platform fact in `PLATFORM.md` gets a pinned
  test with a docstring naming the design argument it protects, so a future mlx
  upgrade fails loudly instead of silently invalidating the tracer or the sandbox.
- **Determinism of the harness itself**: same manifest + seeds + scripted judge must
  produce an identical run log twice (timings excepted).
- **No live-LLM tests.** The real client is covered by one thin contract test behind
  an env flag.

---

## 13. Out of scope for v1 (per the spec)

Precompute (weight-only stretches computed at apply time), reuse (identical stretches
on identical arrays computed once), and cross-job memory are designed in the spec's
"Future additions" and deliberately not built now. The artifact format carries the
reserved, empty `buffers/` directory so precompute can land without a format break;
nothing else anticipates them.

Also out of scope: multi-device, training/backward passes, any dtype or quantization
search (frozen by law), and any per-op GPU counter work (the platform does not offer
one; the region clock exists because of that).
