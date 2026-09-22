# Using an artifact

The job prints how many spots got a proven speedup, the model's time per
step before and after, and the time for a whole sequence of consecutive
steps before and after, both against the baseline the manifest chose
(compiled unless you said otherwise). The sequence comparison is the one
that decides: it times the model the way it is deployed.
`<work-dir>/report.json` has the full account: every spot, why work on it
ended, every attempt with its verdict (each with `floor_ms`, the physical
floor timed beside that kernel), and both step timings, plain and compiled.
`step_ms[workload].win_confirmed` says whether the final measurement resolved
a speedup for an installed replacement. The finished model is also measured
against the baseline the job did not ship against, paired at the end of the job:
under the compiled baseline `step_ms[workload].speedup_vs_plain` (against eager
MLX, compile's own gain included), under the plain baseline `speedup_vs_compiled`
(eager plus its kernels against the untouched model under mx.compile; under 1
means compile alone is faster). `plain_win_confirmed` and its siblings say
whether that comparison resolved. When the final check finds the outputs
right and cannot confirm the win, the job still ends normally: `final.passed` is
false with a `reason`, the session's `outcome` is `unconfirmed`, every measured
number stays in the report, and no artifact is written, since only a confirmed
win ships. Wrong outputs end the job with an error. `step_ms[workload].min_win_ms` is
the least a region win had to save per step there (1% of the step, at most
30 us). `step_ms[workload].steps_per_sample`
is how many dependent steps one timed sample held; more than one means the
step was too short to bring the GPU clock up on its own, and every
whole-model number is still per step. A nominal `speedup` ratio alone can
be noise, especially when nothing was installed. The final timing record
comes from the same comparison as the final whole-model validation.

Before publishing the artifact, the tool loads it in a fresh process and
checks every declared workload and sweep against the patched model, both
through `apply()` and through the bundle's own `load()`. A failed check never
publishes the staged result. The CLI requires a fresh destination and defaults
to `<work-dir>/artifact/`. It rejects paths that overlap the run's logs,
kernels, or checkpoints before starting GPU work.

The artifact folder is a code bundle that needs no optimizer installation:
the model's source copied unchanged, the input
tensors the job measured on (and, for a workload with a `context`, the tokens
that filled the cache, so `load()` rebuilds the same step), the kernels and
generated wrappers, a pinned `requirements.txt`, and a `README.md` written for
that job that states the measured result in plain words and lists which part
of the model got which kernel. It loads, verifies, and re-times itself:

For normal inference, copy `artifact/` into your project, install its
`requirements.txt`, load your chosen compatible weights, then apply the patch.
For an MLX-LM model whose original `build()` returns the model directly:

```python
from mlx_lm import load, generate
from artifact import apply

model, tokenizer = load("/path/to/my-checkpoint")  # or a compatible Hub model ID
model = apply(model)
text = generate(model, tokenizer, prompt="Explain gravity simply.")
```

For custom models, use the original `build()` and weight-loading API before
`apply(model)`. Always use the return value. The operations, module layout,
weight shapes, dtypes, and quantization must remain compatible with the model
that was optimized. If the original builder adds a wrapper, recreate it too.
Different weight values do not require copying weights into the artifact;
architecture or quantization changes require another optimization run. Verify
correctness and speed on the weights and workload you actually deploy.

Alternatively, `from artifact import load` followed by
`load().inference_model` builds from the bundled source and exposes the patched
model for normal inference. This uses `build()`'s original weight selection;
`load()` has no separate checkpoint-path argument. Each artifact's `README.md`
includes both loading routes and a custom-model example.

For workloads with `context`, `load()` is the repeatable benchmark step. Use
`load().inference_model` for normal inference with an advancing caller-owned
cache. A wrapper that includes cache-dependent work falls back to the
original module at untested positions or cache layouts; stateless projection kernels remain
usable as the cache grows. The final sequence measurement includes those
fallbacks.

`build()` retains its original weight-loading behavior: pretrained checkpoints
come from their usual cache, download or local path, and random initialization
remains random. No weights are automatically copied into final artifacts or
recovery packages. `ARTIFACT_FILES` is an explicit opt-in for local resources
that the author wants included; other external paths/settings must be available
where the model runs. Dynamic checkpoint paths and adapter options remain the
builder's responsibility rather than being rewritten by export.

Before loading, the tool checks that model imports can be packaged. External
dependencies need an installed distribution or source in the model project.
Actual MLX-LM snapshots are captured during loading and held fixed for later
comparison builds within the same run. The bundle records those sources and
revisions as provenance, without redirecting its loader to the original
machine's cache paths. Pin a checkpoint revision in the model source when you
need future builds to select that exact revision.

Search finishes each region in ranked order before opening the next. It spends
the full per-region budget, including starter repairs, unless the total budget
is exhausted, the operator requests finalization, or the starter is unsupported.
A win keeps the same search going with its queue and history intact. Remaining
regions are repriced after an installed improvement while search budget remains.

Export writes into a temporary sibling directory and validates there before
publication. A write or validation failure preserves accepted checkpoints and
any existing artifact; a failed publication restores the previous artifact.
The bundle records whether outputs must match exactly or within the manifest
rtol/atol allowance. Validation always compares against the bundled original;
new runs do not require fp32 reference files.
Fresh-process validation builds original and patched models with identical
fresh weights and checks the recorded workloads and cache state. Weight sharing
happens before prefix caches are filled. This supports random initialization
without freezing weights or comparing unrelated random models. Saved test inputs
remain in the bundle for repeatable validation and benchmarking; model weights
do not. Validation uses the existing Hugging Face cache in offline mode, so it
will report missing external resources instead of downloading during export.

```sh
python validate.py     # patched vs original outputs on the saved inputs, the job's own rule
python benchmark.py    # re-time original vs patched the way the final check did; exits 1 unless the win holds
```

When the manifest's baseline was `compiled`, `load()` returns a callable that
already runs under `mx.compile`, so what you time is what the job timed.
Ship it, commit it, or copy it to another machine of the same chip
generation. The code it installs is exactly the code that was measured; to
change the model itself, edit the source and run the optimizer again.
