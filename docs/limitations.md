# Limits and interpreting results

The short version: **only the final, confirmed whole-workload measurement
counts as a speedup.** Everything you see during the search, such as a region's
estimated gain or a kernel that's fast on its own, is a clue the search uses,
not the result.

## What the tool covers

- **MLX on Apple Silicon only.** Other GPU backends aren't supported.
- **Not every part of every model.** Some operations can't be captured,
  rewritten or safely replaced yet. Those regions are skipped, and the report
  lists them with the reason.
- **Cache handling and MLX-LM generation depend on the model's API.** If you
  explicitly ask for something the model doesn't support, the run should refuse
  to start rather than quietly time a different task. See the notes for each
  model in [models/README.md](../models/README.md).

## What a result does and doesn't prove

- **Kernels are verified only for the workloads you declared** (plus any
  runtime checks that apply). At an untested input size or cache layout, the
  model falls back to its original code. One result doesn't prove a speedup at
  every size or on another machine.
- **Small gains can stay unconfirmed.** Timing noise, background GPU work, heat
  and power state all affect measurements. When the tool can't separate a small
  gain from noise, it doesn't ship it. More attempts don't guarantee a
  confirmed win.

## Using the result

- **Artifacts don't include weights by default.** They need a compatible
  architecture, dtypes and quantization, the same dependencies, and whatever
  external files your `build()` loads. See
  [artifact compatibility](artifacts.md#apply-it-to-your-model).

## Running a job

- **The budget counts attempts, not time.** A run's length depends on the
  model, and can be hours.
- **Jobs can stop early** on a worker failure or a timeout. If that happens,
  keep the report, the console output and the checkpoints.
