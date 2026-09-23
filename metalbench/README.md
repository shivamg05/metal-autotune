# MetalBench for autotune

**Latest published run:** [Apple M4, 19 of 35 workloads faster than compiled MLX](published/2026-09-19-m4/README.md)

This is a KernelBench-style benchmark for the tool. It runs the optimizer over
a fixed set of problems and scores how much faster each one got.

**Where the problems come from.** They're the problem definitions from
[MetalBench](https://github.com/Lazarus-931/MetalBench) (MIT), vendored at the
commit in `problems/PIN`. There are 116 MLX modules in three sets, each with
registered input shapes and tolerances:

- `common`: single operations
- `standard`: fused pairs of operations
- `full`: small complete models

**Only the problems are borrowed, not MetalBench's scoring.** Its harness
divides GPU-only kernel time by the wall-clock time of the MLX version. That
counts MLX's dispatch overhead as a "speedup" the kernel didn't actually earn.
Here, each problem is timed as a whole workload instead.

## How it works

Each problem becomes an ordinary optimization job. A bridge generates a model
file and a manifest from the problem's shapes and tolerances. The normal
harness then checks correctness and compares the whole workload against the
baseline you pick.

**Pick the baseline deliberately.** The CLI defaults to `--baseline plain`:
a kernel ships if it beats eager (uncompiled) MLX, which is the bar KernelBench
and similar benchmarks use. `--baseline compiled` makes a kernel beat the model
under `mx.compile`, which is the harder bar. The published results use compiled.

Eager runs are saved as `<chip>.eager[.tag].json`, so the two kinds never mix.
Either way, both columns get measured: at the end of every job, the finished
model is also timed against the other baseline.

## Reading the scoreboard

- The scoreboard shows speedups against both compiled and eager MLX.
- A problem with nothing shipped scores **1.0** against its baseline. That
  includes problems the tool can't handle at all (e.g. no starter kernel for
  an operation). They count against it, which is the point of a benchmark.
- The **kernel** column shows how the best kernel did on its own region. That's
  search evidence, not necessarily a gain in the installed model.
- **`fast_p`** is the fraction of all problems that got strictly faster than
  `p`. A failed job counts as not faster.
- **Only compare like with like:** same baseline, shapes, judge settings and
  budgets. A gain over eager MLX doesn't show a gain over compiled MLX.

## Running it

```bash
uv run python metalbench/fetch.py                       # once, or to re-vendor
uv run python metalbench/run.py --set standard --judge claude-cli
uv run python metalbench/run.py --only abs,rms_norm_linear --budget-per-region 4 --budget-total 8
```

Without flags, `run.py` runs every set against the plain baseline, with 4
attempts per region and 8 per problem. The exact command behind the published
results is in [its README](published/2026-09-19-m4/README.md).

## Where output goes

Everything lands under `runs/metalbench/`, which Git ignores:

- `results/`: the scoreboards, kept per chip. Each row includes the job's
  `work_dir`.
- `jobs/<timestamp>/`: generated manifests, logs, reports and artifacts for
  each problem.

Every invocation gets a new `jobs/` folder, `--rerun` included, so earlier
kernels and logs are never overwritten. Use `--output-dir /path/outside/the/repo`
to put everything somewhere else. Output in the old layout is left where it
is, because it can contain absolute paths.
