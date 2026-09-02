# Implementation Plan: Autonomous MLX Kernel Optimization

This plan turns `DESIGN_SPEC.html` into a working tool. Read the spec end to end first.
The spec is the authority on WHAT the system does and its vocabulary (manifest, workloads,
trace, regions, roofline, scaffold, hypothesis queue, the ladder, bind, the artifact).
This plan is the authority on HOW to build it: the stack, the module boundaries, the data
formats, the build order, the platform facts you must verify before trusting them, and the
decisions the spec leaves open. Where this plan and the spec disagree, the spec wins.

One piece of history worth knowing: the spec's bind originally installed a runtime
interception wrapper on the op stream. On 2026-08-30, after surveying how other
kernel agents deliver kernels, the owner changed both documents to the current
mechanism: generated replay wrappers installed by module swap. Section 5.2 records
the mechanics and the reasoning. Spec and plan now agree on bind.

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
model by swapping in a generated wrapper module that calls it; a retrace and an
end-to-end check confirm the bind is real; only then it ships. The job ends with an
artifact (kernels, a generated patch module, a swap table, an `apply()` installer, a
report) that speeds up a freshly loaded model without any of the search machinery.

Three components, kept strictly apart:

- **The harness**: ordinary code. Owns tracing, region cutting, pricing, all measurement,
  all correctness gates, bind, and the artifact. It never judges a kernel by taste, only
  by measurement.
- **The judge**: an LLM behind a narrow JSON boundary. Sees metadata only (shapes, dtypes,
  verdicts, roofline distance, kernel source it is editing). Never sees tensors, weights,
  or activations. Never overrides a failed check. Proposes one hypothesis at a time and
  writes one Metal kernel body for the front item of its queue.
- **The artifact runtime**: a tiny package that ships inside the artifact: `apply()`,
  kernel loading, and the module-swap installer. The wrappers themselves are generated
  per job as readable Python with explicit kernel calls, and the harness installs the
  same generated code it measured, so the thing you measure is the thing you ship.

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
- The baseline is chosen empirically at job start: measure the step clock both ways,
  the model exactly as `build()` hands it and the same callable under harness-applied
  `mx.compile`, paired and interleaved per Section 6, and the faster one is the
  baseline every later number is scored against. Record both clocks and the choice in
  the report. Tracing always records the plain model (a fully compiled model records
  as one opaque call and yields no regions); the compile choice affects only what the
  clocks and e2e compare against. If the compiled baseline wins, law 10 applies to it:
  a freshly compiled callable after every bind, before any timed pass. (Motivating
  history: `mx.compile` once measured ~4% slower than plain on a small test net, so
  neither outcome is safe to assume; the M0 spike measures both on the pinned version.
  If the model's own code compiles internally, that is the model's business and the
  tracer treats compiled sections as opaque calls.) Known cost of a compiled baseline:
  pricing replays plain ops, so a region `mx.compile` already fuses can price as
  recoverable and then die at the e2e veto. That wastes attempts; it never falsely
  ships.
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
                     # T_mem/T_compute/T_launch, bound, s_max; the bytes-and-
                     # launch floor comes measured from measure/probe.py; the
                     # whole step's floor (the scout line)
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
    emit.py          # generate replay-wrapper classes from the trace (Section 5.2)
    certify.py       # static replayability screen + identity certification
    swap.py          # module-tree swap install/uninstall (thin over runtime.swap)
    verify.py        # literal retrace check (Section 7.10)
  e2e.py             # orig-vs-orig floor, patched-vs-original (both assoc paths),
                     # step-time veto (Section 7.11)
  loop.py            # region open/close state machine, close rules, budgets
  artifact/
    emit.py          # write kernels/, patch/, swap_table, report; package the runtime
  report.py          # report.json schema and writers
  log.py             # append-only run log, sign conventions
  cli.py             # `autotune run manifest.yaml`

autotuner_runtime/   # tiny package, no imports from autotuner/
  kernels.py         # rebuild mx.fast.metal_kernel objects from .metal + launch.json
  swap.py            # resolve scope addresses on a fresh model, install/uninstall
                     # wrapper instances, fallback-engagement logging
  apply.py           # load swap_table + generated patch module + kernels, swap onto
                     # a freshly loaded build() model

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
roofline: {T_mem, T_compute, T_launch, T_floor, T_roofline, s_max}}`, plus `fingerprint`
(Section 5.5), `sweep_instances` (per sweep size: the matched node span, shapes, and
scalar_args at that size; Section 5.6), and `boundary_store` (paths to saved
input/output tensor sets, k per workload; Section 5.7).

**Hypothesis / queue item**: `{id, kind, assoc_tag: preserving|changing,
family_id: str, hypothesis: str, depends_on: id?, condition: correct|shipped|failed}`.
`family_id` is the judge's own label for the algorithm family an item belongs to
(one-pass vs two-pass attention, split-K, ...), so it can drop a whole family in one
batch of deletes; the harness reads nothing into it. Head and shipped are the spec's two
per-region bookmarks. Only the front ready item is ever turned into Metal.

**KernelAttempt**: `{hypothesis_id, parent_kernel_id, source: str, header: str,
launch: {grid, threadgroup, template}, fallback_predicate?: str}`. `grid`,
`threadgroup`, and `fallback_predicate` are expressions in the launch grammar
(Section 5.8), evaluated by the harness against actual shapes at every call, so one
kernel can launch correctly at every sweep size. Every attempt is stored whether it
fails or ships; failed kernels are legal parents.

**Verdict**: `{hypothesis_id, outcome: failed|correct_slower|tentative_ship|shipped|
rolled_back, failed_gate?: str, gate_detail?: {...}, region_ms?: float,
samples?: [...], e2e?: {...}}`.

**SwapEntry**: `{scope_address, wrapper_class: str (a name in the generated patch
module), regions: [{region_fingerprint, copies: [member node span per copy],
shape_dispatch: [(shape_predicate, kernel_id)], fallback: replay of the original op
sequence}]}`. One entry per outermost shipped scope: every shipped region inside that
scope splices into the scope's one generated wrapper (a wrapper replays its scope's
whole recorded stream, so nested or sibling wrappers inside it would never run).
Addresses are serialization only: at install time (harness bind or artifact
`apply()`), each scope address is resolved against the concrete model to a live module
instance and its parent, and the swap happens on instances (Section 5.3).

**Artifact** on disk:

```
artifact/
  kernels/<id>.metal + <id>.launch.json
  patch/wrappers.py        # generated replay-wrapper classes: readable Python,
                           # explicit kernel calls, fallback path included
  swap_table.json          # [SwapEntry]
  buffers/                 # reserved, empty in v1 (future precompute)
  runtime/                 # the autotuner_runtime package, vendored
  apply.py                 # thin shim calling runtime.apply
  report.json
```

(This matches the spec's artifact block: kernels, wrappers, swap_table, apply(),
report.)

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
  would not help, because the cache's reference points at a lazy array nothing
  installed later can reach. Both checks run; disagreement resolves toward retained.
- `mx.compile` wrapping: a model that compiles part of itself gets that part recorded as
  one opaque call (op = `compiled_fn`, inputs and outputs recorded). Opaque calls can
  never be inside a region, but they anchor completeness.
- The tracer installs before the model file is imported, because a model file that did
  `from mlx.core import matmul` at import time would keep the unwrapped function.
  The CLI enforces the ordering; the tracer refuses to install if the model module is
  already in `sys.modules`.

### 5.2 Bind by generated replay wrappers (the delivery mechanism)

**Design note.** The spec's bind originally installed a runtime interception wrapper
on the op stream (arm at the last op, a substitution map redirecting consumers). By
owner decision (2026-08-30) both documents now use this mechanism instead: the
harness GENERATES Python source, wrapper classes whose `__call__` replays the
enclosing module's recorded op sequence with the region's ops replaced by the kernel
call, and installs them by swapping module instances on a freshly loaded model. This
is the strategy other kernel agents use for model composition, done mechanically from
the trace instead of by an LLM. The model's own source is still never parsed or
modified; the generated code lives in the artifact, not in the model file. Everything
bind must guarantee is unchanged: retrace proof, shape dispatch, library fallback,
e2e promotion, an artifact that works without the harness.

Why the change: zero runtime overhead (no global hooks anywhere, nothing intercepted
at inference time), the artifact becomes readable Python with explicit kernel calls
(the way every human integrates a kernel), and the spec's retrace check "the cut
became one custom dispatch" becomes LITERALLY checkable, because the member ops are
actually gone from the patched stream instead of present-but-dead.

Mechanics:

- **Delivery scope.** For each region, the scope is the smallest module (by enclosing
  address) containing all its member ops; the top-level callable is the outermost
  scope. The wrapper class for a scope is generated from the trace via the replay
  name table (Section 5.4): it re-executes the scope's recorded ops in order, with
  each shipped region's span replaced by the kernel invocation, shape dispatch
  evaluated through the launch grammar, and the fallback path being the original op
  sequence itself. Both paths live in the generated code, so fallback needs no
  library magic. Generated source calls ops by live `mx.` attribute lookup
  (patch-visible, so retraces record it) and calls kernels through a small runtime
  shim, `autotuner_runtime.kernels.call`, which record mode also patches so the
  custom dispatch appears as a node in retraces. Each wrapper also carries a span
  map (which emitted op came from which baseline address and position); retrace
  consumers align the flattened stream through it (see 7.10).
- **Weights and state.** The wrapper holds a reference to the original module it
  replaced and delegates attribute access to it, so model code reaching through
  children keeps working and the weights used are the live model's own arrays, no
  transplant. (M0 spikes 04/05: wrapper classes must subclass nn.Module, because
  Module.__setattr__ drops a plain-object child from the module tree and the subtree
  vanishes from parameters()/named_modules(); and the delegating __getattr__ must
  raise AttributeError, not KeyError, before its wrapped attribute is set, or
  Module.__setattr__'s hasattr probe breaks during __init__.) At freeze time, weight `array_id`s are resolved to parameter PATHS via
  the model-attribute snapshot walk Section 5.1 already performs; generated code reads
  each weight by path through the wrapped original, which is what makes the same
  generated source valid on any fresh `build()` model. Scalar args recorded inside
  the scope must be constants across the scope's recorded calls or derivable from the
  scope's own call arguments (a decode step's rope offset varies per call and must
  flow from the argument, not the recorded constant); the generator enforces this and
  certification backstops it.
- **State calls.** A plain Python object reachable from the model that holds
  arrays (mlx_lm's KVCache, a namespace of buffers) is a state holder. While
  recording, its methods are wrapped like module calls: the call runs with
  recording suppressed and records one opaque node, `state:<Class>.<method>`, with
  the object's identity and model path. The generated wrapper replays it as the
  same call on the same object, reached through the scope's call arguments (a
  cache handed down the tree, found by identity, even inside a list) or through
  the wrapped module by path, so the cache write and whatever else the method does
  to its state happen for real, where the recorder could never see them. A state
  call is a chain barrier, never inside a region, and never replays in process.
  Scalars the model reads off the object (a rope offset) are recorded as the
  constants they were, which certification over repeated calls checks. Before this
  (2026-09-02) every fusion inside an attention block stranded on the KV-cache
  write: 67 of 82 candidates on the Qwen3 decode job, 66 of 73 viable after.
- **Static replayability screen** (at region build): a scope qualifies only if its
  recorded stream has no opaque compiled calls ANYWHERE in the scope (the wrapper
  replays the whole scope, and an opaque call has no serializable callable to
  re-invoke), no in-pass evaluation (data-dependent control flow), no
  `python_retained` productions (a slice write straight into a kept array, not
  through a method), scalar args passing the rule above, and one stream
  that matches, node for node, everywhere the scope fires: every workload and every
  sweep retrace. A scope whose stream differs anywhere it fires is branchy and fails.
  A region with no qualifying enclosing scope is rejected with reason "no certified
  delivery scope". That is this plan's concrete meaning for the spec's rejection rule
  "there is no place to install a wrapper on it".
- **Identity certification** (once per scope, off-clock, before its first ship; the
  loop runs it between gate 9 and retrace on the first ship into a scope, and owns
  the escalate-to-parent-or-strand handling): swap in an IDENTITY wrapper, one that
  replays the scope's recorded ops with no kernel change. It must be invisible: the
  retrace stream matches baseline under the span-map projection, outputs match
  bitwise on every workload and sweep size, and repeated calls of a stateful step
  match call by call. Replaying the same math must be a no-op; any divergence means
  the scope is not actually replayable, and the region escalates to the parent scope
  or strands. This one control converts "we believe the trace is faithful here" into a
  measured fact.
- **Wrapper versioning.** A scope has one wrapper covering all its shipped splices,
  always generated around the PRISTINE original module. A new ship into an occupied
  scope regenerates the wrapper with all current splices and replaces the occupant;
  rollback restores the previous occupant, not the pristine module; an outer scope
  shipping over an inner wrapper retires the inner one and merges its splices into
  the outer wrapper.
- **What this strands, on purpose.** Regions whose every enclosing scope fails the
  screen: a region overlapping a kept array written without a method, top-level
  glue inside a stateful decode step, anything under data-dependent control flow
  at every available scope.
  Stranded regions are reported with their p and the failing reason, never silently
  dropped. If real runs show material wins stranded, the known escalation is the
  runtime-interception layer this section replaced (hook the op surface, arm at the
  last op, substitute consumers); it was cut for complexity and overhead, and would
  return as a fallback tier only for stranded regions.

### 5.3 Scope addresses and install

Addresses are strings for serialization; install always operates on live objects.
`apply()` and harness bind walk the fresh model, resolve each SwapEntry's
`scope_address` to a module instance and its parent (erroring loudly on a miss, since
an earlier shipped change or a model edit can restructure the tree), instantiate the
generated wrapper around the original instance, and swap it in by parent attribute
assignment (platform fact, re-verify in M0: `mlx.nn.Module.update_modules` rejects
numeric path segments as dict keys and needs the list-form update spec; plain parent
`setattr` works for named children). Uninstall is the reverse swap; rollback is
uninstall. Two model instances in one process never collide because each install
touches only its own model's tree, which is what makes the interleaved e2e arms sound
with no further machinery.

### 5.4 Trace replay

`trace/replay.py` is the one module that re-executes a recorded node sequence, and
four consumers depend on it: region-clock pricing (run the region's ops on saved
inputs), the fp32 golden (same ops, promoted dtypes), sweep reference generation
(library outputs at swept sizes), and the wrapper generator (`bind/emit.py` shares
the same op NAME table, so replayed-in-process and generated-as-source are one
semantics; but generated source calls ops via live `mx.` attribute lookup so retraces
can see it, while in-process replay binds saved originals and stays patch-invisible).
Contract: the op-string-to-callable resolver is a standalone
table built by importing `mx` and resolving module attribute paths and `mx.array`
dunder/method names (a slice read replays as `mx.array.__getitem__`); the patch
installer consumes this same table rather than defining it, so replay works in a
patch-free subprocess where no tracer was ever installed (which is where gates 5, 8,
and 9 use it). Replay folds over nodes in `seq` order,
binding `array_id`s to live arrays (starting from provided bindings for region inputs
and weights), invoking the saved originals with `scalar_args` verbatim, and returns the
arrays for requested output ids. Hooks: a dtype-promotion transform and an op
substitution table (for the golden, Section 7 gate 8). Replay never goes through the patch
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
be one kernel correct everywhere and faster everywhere. Weights count by role, dtype,
and shape, never by value: a projection against a different weight shape is a
different kernel to write, price, and check. The fingerprint is a hash of the canonical form. Cross-sweep-size
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
region as `sweep_instances`. A scope whose stream diverges at some sweep size (a
branch went the other way) fails the replayability screen (Section 5.2), so its
regions strand before any kernel work is spent; divergence discovered here is
reported as the screen's evidence. Boundary inputs and library references at swept
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
flattener that inlines each entry point into a self-contained header. (M0 spike_08:
feasible, a flattened steel GEMM header compiles; the flattener must skip the
auto-prepended prelude, utils.h and its transitive includes, or hit redefinition
errors, and must terminate output with a newline or a trailing line comment swallows
the generated signature.) Build the flattener; stitch wherever the definition above
holds; fall back to naive lowering everywhere else.

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
  call. The harness must eval a probe output under try/except to detect a broken build.
  (M0 spike_06: reported line numbers count the full concatenated program, utils.h plus
  the generated signature plus the body, and the offset GROWS with the generated
  signature (472 for 1-in/1-out, 473 for 2 inputs, more with shape buffers or grid
  built-ins), so the harness computes the offset per kernel rather than subtracting a
  constant. spike_04: the kernel name is pasted into the generated signature, so a
  non-C-identifier name breaks compilation with the error surfacing only at probe eval;
  static checks validate the name. For transcendentals, library-bit fidelity comes from
  metal::precise:: namespacing, not math_mode.)
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
  consumes hypothesis budget (a babbling judge must exhaust its region, not stall it).
  A reply with nothing to evaluate (a yield, a batch of plan edits the queue refuses,
  a kernel for an item that is not ready) is handled the same way one level up: the
  loop asks again once with the reason, then each further one costs an attempt.
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
| Watchdog | 20x library region time (raised from 10x by maintainer decision 2026-08-31: correct fused-chain starting kernels sit near 10x, and the gate exists to catch wedged kernels, not honest slowness) |
| Ship margin | max(1% of the library region time re-measured in this verdict, 3 sigma of the interleaved samples) |
| Region close | the region's budget or the job's is spent, nothing else (maintainer decision 2026-09-02: the plateau, streak, roofline, and family rules closed regions the judge would have kept improving, and a yield is refused while budget remains) |
| Scaffold fix attempts | 1 |
| Determinism runs at gate 8 | 3 |

Plan defaults (tunable, recorded):

| Constant | Default | Spec language |
|---|---|---|
| Region floor | 2% of step (copies combined) | "about 2%" |
| Roofline has_room | s_max >= 1.2, from the measured floor at pricing and again from the sandbox's clock when the region opens | "barely beats ... is skipped" |
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
   (M0 spike). (M0 spike_10 refinement: after a pacing idle the GPU runs at ramped-down
   clocks and the first sample reads 1.5-1.6x steady state, so pacing is chunk-granular
   and every chunk takes ~2 unmeasured ramp-warm samples of the functions about to be
   timed before any measured sample; per-sample pacing biases every pair and inflates
   the A/A floor ~50x.)
3. **Warm until stable, not a fixed count.** Warm every distinct shape once, then repeat
   until two consecutive timings agree within 1% (cap ~30). Fixed small warmup counts
   have left first-ever kernels reading 2x slow (pipeline JIT plus page faults;
   M0 spike pins convergence on a first-ever kernel). The spec's rule that the first
   eval of each workload is thrown away is the floor, not the ceiling. (M0 spike_10:
   a first-ever kernel's first call measured 63x steady state, dominated by ~90ms of
   Metal compile; and the 1% agreement rule needs an absolute epsilon floor, default
   100us, because 1% of a ~1ms kernel is under dispatch jitter and never converges.)
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
    Any compiled callable, whether a harness-compiled baseline (Section 2) or a model
    whose own forward compiles internally, must be freshly built after every bind
    before any timed pass. (M0 spike_09 pinned the caching behavior and found the
    remedy must be stronger than "call mx.compile again": the cache entry survives
    while any old compiled object is alive, so recompiling the same function object
    still returns the stale graph. The harness therefore compiles a newly defined
    closure over the model after every swap; dropping every old compiled object first
    also works but is fragile.)
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

13. **Every timed loop reads from memory and runs one pass at a time.** Metal runs
    independent launches side by side, so a loop of unchained passes measures throughput
    on a full GPU, not the latency a model step pays, where each layer waits for the
    last; a one-threadgroup kernel read ten times faster than it ran in place. And a
    few saved input sets sit in the cache, which a step's weight stream never does.
    So the pricing clock and the ship clock rotate a working set past the cache-defeat
    threshold and chain their passes: each pass's smallest non-weight float input
    carries a zero from the previous pass's first output. A per-pass figure comes from
    pairing the loop against the chain alone, which takes out the link's cost and the
    sample's fixed submit-and-sync cost alike; a win is the paired difference between
    the two arms, which pay the link equally.
14. **A limit that decides is measured beside the thing it limits.** The bytes-and-
    launch part of a region's roofline is a probe: one launch that streams the
    boundary bytes (`measure/probe.py`), run in the same chained, rotated loop as the
    region and paired against it in one window, at pricing and again beside every
    kernel in the sandbox (`floor_ms`). Region over probe is then a ratio the
    machine's speed cannot move. The arithmetic limit was wrong two ways at decode
    sizes: a dependent kernel pays its launch and its stream in series, where
    `max(T_mem, T_launch)` assumed overlap, and the peak from a 512 MB pass is out
    of reach for a 2 MB weight; and dividing a clock from one minute of a job by a
    peak from another read 15% of machine drift as headroom (the 10:54 Qwen run:
    1.29 to 1.40x priced, 1.05 to 1.17x at open). The flops term stays arithmetic
    against the matmul peak, so a compute-bound region's ceiling is MLX's own GEMM.
    The whole step gets the same accounting once, as the scout line: outside bytes,
    flops, launches, floor, and room, so a manifest with no room says so before
    any search.

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
   the child, the watchdog factor (Section 5.12) compares the kernel's first timed run
   against a library
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

With replay-wrapper delivery, the spec's check is literal. Generated wrappers are
ordinary Python calling live `mx.` attributes and the recorded kernel-call shim, so a
record-mode retrace of the patched model sees exactly what runs. One alignment detail:
a wrapper replays its scope's ops inside its own `__call__`, so ops the baseline
recorded under nested child addresses now attribute to the scope address. Matching
therefore runs under the wrapper's span map (Section 5.2): each replayed node is
projected back to its baseline (address, position) before comparison, for both
verification here and identity certification. `bind/verify.py` checks the frozen
retrace:

- the region's member ops are GONE from the stream, replaced by one custom-kernel node
  per copy;
- that node consumes the cut's inputs, and its outputs feed every downstream consumer
  and live value the region's outputs previously fed;
- under the span-map projection, the node stream outside the cut matches the baseline
  trace on (op, specs, projected address): neighbors unchanged.

Identity certification (Section 5.2) has already proven the scope's replay is
invisible with no kernel change, so any retrace miss here is attributable to the
kernel splice itself. Any miss is a failed bind: swap the original module back, tell
the judge.

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
  ORIGINAL model under it, once, to produce the model-scale fp32 golden. The
  interceptor always runs the PLAIN form of the original, whatever the baseline
  choice, because the patch surface cannot see inside a compiled callable. Both arms are
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
- Both arms live in one process, weights shared (law 9). The baseline arm is a fresh
  untouched `build()`; the patched arm is a fresh `build()` with the generated
  wrappers swapped into its tree. Under a compiled baseline (Section 2), both arms
  are compiled: a fresh `mx.compile` of the untouched model, and a fresh `mx.compile`
  of the swapped model built after install, per law 10. No hooks exist in either arm,
  so there is no overhead-charging machinery to get right: the patched arm's only
  added cost is the generated Python it actually runs, inherently included in its own
  timing. M8's acceptance includes proving the baseline arm's retrace fires zero
  custom dispatches.

Failing bind or e2e rolls the ship back and the judge mutates the queue.

---

## 8. Subprocess execution model

Every kernel evaluation is out of process. One `sandbox/worker.py`, three launch modes,
selected by environment at spawn:

- **validate mode**: `MTL_SHADER_VALIDATION=1` plus `MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`.
  Runs gates 1-8. (M0 spike_07: validation alone reports to os_log only and zerofills
  invalid accesses without faulting, so the harness gets no signal; with the stderr
  variable set at launch, "Invalid device load"/"Invalid device store" lines appear on
  child stderr with kernel name and offset, and the worker parses them into gate detail.)
- **score mode**: clean env. Re-runs smoke + determinism, then gate 9. Also used for
  the step clock, region pricing, certification and bind retraces, and e2e once M5
  lands (M3/M4 may measure in-process with an asserted-clean environment; M5 migrates
  them and checks the numbers agree).
- **capture mode** (debug only, off the hot path): `MTL_CAPTURE_ENABLED=1`, wraps a
  single dispatch in `mx.metal.start_capture` for humans to replay in Xcode.

Protocol: parent writes one JSON job spec (manifest ref, region fingerprint, boundary
store paths, kernel source + launch expressions, gates to run, seed; for
certification, retrace, and e2e jobs also the generated patch-module source and the
swap entries to install) to the child's
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
- Delivery micro-spike: hand-write a replay wrapper for a two-child toy module (its
  recorded ops re-executed in order, with a hand-fused kernel spliced over two of
  them), swap it in by parent attribute assignment, and verify outputs match, a
  retrace shows the member ops gone, and the identity form (no kernel, pure replay) is
  bitwise invisible, including across repeated calls of a stateful toy. This is the
  whole bind idea in fifty lines; if it does not work, stop and redesign before M1.
- Module swap semantics: parent `setattr` swaps a named child; `update_modules`
  rejects numeric path segments and needs the list-form spec; a wrapper delegating
  `__getattr__` to the wrapped module survives model code reaching through children.
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
  plain-vs-compiled step time on a small test net (a dry run of the per-job baseline
  choice in Section 2).
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

Builder (singletons, chain growth, which stops before a matmul-like op that reads
a matmul's output because a dependent matmul is a second kernel rather than a fused
one, view absorption, slice-write termination, the four
rejection rules, per-stretch liveness derivation), fingerprint grouping, the static
replayability screen assigning each candidate its delivery scope (the "no certified
delivery scope" rejection lands here; dynamic certification waits for M8), batched
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
cheats (the live-output sabotage variant and the wrong-scope swap entry) wait for M8,
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

Wrapper generation from traces (`bind/emit.py`), the static replayability screen,
identity certification, module-swap install and uninstall, the swap-table format, the
literal retrace verification, e2e floor and both assoc paths and the veto, rollback
(swap the originals back).

This is the riskiest milestone after the M0 spike; do not let it slip. Done when: a
hand-written correct kernel for a fixture region ships through a generated wrapper;
retrace shows the member ops gone, one custom dispatch per copy, neighbors unchanged;
identity certification passes on replayable fixtures and correctly REJECTS the
stateful-scope and branchy-scope fixtures, whose regions strand with named reasons;
e2e sits on the orig-vs-orig floor with step time unchanged or better; the fixture
whose region output is also a step output gets the kernel's output from the wrapper's
return; with both arms live in one process, the baseline arm's retrace fires zero
custom dispatches; an assoc-changing hand kernel passes e2e only under the
golden-relative path while an assoc-preserving one sits on the floor; a stateful
decode fixture with a per-call scalar (a rope-offset stand-in) certifies only when the
scalar flows from the scope's call argument; a second ship into an occupied scope
regenerates one wrapper with both splices, and rolling back the second ship restores
the first ship's wrapper, not the pristine module; sabotage cases roll back cleanly
(kernel writing a wrong live output fails e2e; a swap entry naming the wrong scope
fails retrace); the runtime package plus the generated patch module alone (no
`autotuner` import) applies to a fresh `build()` model and reproduces the patched
outputs.

### M9: Judge (`judge/`)

Schema, prompts, scripted judge, real client, the queue mechanics (pop-ready,
depends_on conditions, mutations after verdicts applied as one batch).

Done when: the full hypothesis cycle runs against the scripted judge deterministically
(seed queue, execute front item, verdict, mutate); prompt rendering is snapshot-tested;
a malformed LLM response burns one re-ask then one hypothesis, never the run; a batch
of plan edits lands whole or not at all.

### M10: The region loop (`loop.py`)

Open (scaffold, fix-or-skip), the hypothesis cycle wired to sandbox + ladder + bind
(including identity certification on the first ship into each scope, with the loop
owning escalate-to-parent-or-strand when certification fails), the budget as the
one close rule, retrace-after-close with share updates and covered-region dropping
(through span maps on patched models), budgets, the final e2e plus the sweep-fallback
pass through the real installed wrappers, full run logging.

Done when: with the scripted judge on the planted-win fixture, the whole job runs
end to end: finds the region, ships a scripted winning kernel through bind + e2e,
closes when its budget is spent, retraces, and the run log replays the spec's
example history shape (fail, fix, climb, ship, close on budget). A second scripted run on the vendor-parity
fixture ships nothing and says so.

### M11: Artifact (`artifact/`)

Emit kernels + the generated patch module + swap_table + vendored runtime + report;
`apply()` in a fresh process.

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
budget: attempts left for the region and for the job
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
  covered by a shipped larger cut, per the spec; the log records both. On a patched
  model this retrace locates remaining regions through the installed wrappers' span
  maps (the same address projection as 7.10), so regions living inside an
  already-shipped scope stay findable.

---

## 12. Test strategy

- **Fixture zoo** (`tests/fixtures/`): tiny models, each existing to exercise one
  mechanism. Operator soup (dunders, reflected, in-place, slicing, slice-write);
  norm + three projections sharing an input (chain growth without data edges);
  views-only stretch; mid-stretch Python retention (a cache object, retained AND
  consumed); a region output that is also a step output; a plain-function model (no
  nn.Module anywhere); an eager model calling `.item()` mid-forward; data-dependent
  branch; compiled submodule; repeated identical layers (copy grouping); a
  stateful-scope model (cache mutated inside the scope) and a branchy-scope model,
  both of which must fail certification and strand their regions with named reasons;
  a stateful decode toy whose per-call scalar must flow from the call argument; a
  planted-win model whose chain a naive fusion beats; a vendor-parity model. Fixtures
  are also the M12 controls at larger scale.
- **Cheat zoo** (`tests/cheats/`): the harness's job is rejecting bad kernels, so its
  tests ARE bad kernels, each asserting WHICH gate catches it: partial-write (gate 3
  poison via init_value); shape-hardcoded (gate 7 sweep); stride-lying (gate 7
  transposed input); cached-by-shape (gate 5 k-set value variation);
  cached-by-shape-and-magnitude (gate 5 value regimes, which change magnitudes);
  fp16-sloppy accumulation (gate 5 regimes: large-scale and outlier inputs);
  eps-dropping (gate 5 tiny-scale regime); atomic-racy (gate 8 determinism);
  live-output-dropping (gate 1 static, and e2e for the sabotage variant);
  fallback-missing (gate 1); fallback-declared-but-dead (gate 7 fallback log);
  wrong-scope swap entry (retrace verification).
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
