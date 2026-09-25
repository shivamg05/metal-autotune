# Architecture

The tool optimizes a declared workload on a particular machine. The AI proposes
code; the harness independently decides whether that code is correct and faster.
The invariants below describe the boundaries that implementation changes must preserve.

1. **Load and trace.** `manifest.py` defines the model, inputs, budget and measurement
   objective. `trace/` records the operations and values crossing their boundaries.
2. **Select regions.** `regions/` groups legal replacement regions and estimates
   where replacing work could save whole-model time. Estimates guide search;
   they never establish a speedup.
3. **Build a starter.** `scaffold/` uses supported lowering, MLX source, or captured
   custom-kernel source to provide an editable implementation. If lowering cannot
   produce a starter, a replayable region keeps its original calls as the reference;
   the agent can write replacement Metal from scratch, in one or multiple stages.
   The reference itself cannot ship. Unreplayable or unbindable regions stay unsupported.
   If a region appears at several shapes, its reference retains the original wiring
   for each shape. One proposed kernel must pass all recorded cases; other shapes
   still use the original model. A failed generated starter takes this same path
   before spending a judge attempt on repair.
4. **Search.** `judge/` supplies bounded metadata and code context. The agent plans
   four opening designs, then may revisit earlier candidates, combine ideas, or
   start another design. Budgets limit actual attempts. Full earlier code remains
   readable; earlier successes and failures inform later proposals.
5. **Verify and measure.** `ladder/`, `sandbox/`, and `measure/` check candidates in
   workers. `bind/` and `loop.py` verify installation and compare the actual model
   with its baseline. Bad outputs, ineffective replacements and unresolved gains
   do not become confirmed wins.
6. **Export.** `artifact/` writes a bundle, validates it in a fresh process, then
   publishes it. `autotuner_runtime/` runs the installed kernels without depending
   on the search harness. See [artifact usage](artifacts.md).

## Installation and exported baselines

Graph matching constrains known boundary inputs (module weights and arguments)
before selecting branches. Distinct recorded operations must match distinct live
operations; boundary inputs may still alias. Before searching a region, an
original-computation replacement checks its graph cuts across the declared
workloads, then restores the incumbent. A failed check closes the region without
spending judge attempts. Candidate correctness and performance gates still apply.

Library-inference bundles retain the exact empty compiled wrappers used by the
job's baseline. Artifact benchmarks reinstall those wrappers on the original arm;
correctness validation continues to use the untouched model. Older bundles lacking
that baseline cannot claim a compiled comparison and must be re-exported.

## Boundaries to preserve

- Keep model precision, quantization and intended computation unchanged.
- The judge receives approved metadata and source, not tensors, weights,
  activations or numeric acceptance thresholds. It cannot override a gate.
- Arithmetic-preserving candidates must match exactly. Arithmetic-reordering
  candidates use the manifest's tolerance policy. Tolerance is not permission to
  change the model or its number format.
- Compare like with like: the same inputs, cache state, execution mode and objective.
  The recorded baseline is part of the result. Both compilation and replacement
  overhead must be represented in the deployed comparison.
- Compilation-time graph substitution is preferred where safe. Other supported
  scopes use certified replay. The implementation exported must be the one tested.
- Pair measurements, warm the GPU, pace GPU work, and account for uncertainty.
  Force lazy work to finish inside the measured interval. A timeout is an execution
  failure, not a performance verdict. Workers share the physical GPU.
- Cache state must be reset or advanced consistently with the declared task.
  A forward pass and a full generation request are different measurement objectives.
- Validate every live output and the whole model, then validate the exported bundle
  in a fresh process. Preserve accepted checkpoints if finalization fails.

Behavioral platform assumptions are pinned by the `test_platform_*` tests and
other regression tests. See the [measurement reference](measurement.md) for
log fields and timing details. Reusable timing investigations live in `tools/`; old
machine-specific measurements are not universal constants or product guarantees.

## Development

Run `uv sync`, then tests relevant to the change, for example
`uv run pytest tests/test_judge.py tests/test_widening.py`. The full suite includes
GPU jobs and can take substantial time. Run GPU tests serially, without another
optimization active. Live-provider tests are opt-in.

Pull requests are welcome. Keep changes focused, preserve the invariants above,
and include the relevant validation. Keep generated runs and weights out of Git.
