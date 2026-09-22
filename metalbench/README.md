# MetalBench for autotune

[Published Apple M4 results: 19/35 wins, up to 2.34× over compiled MLX](published/2026-09-19-m4/README.md)

A KernelBench-style benchmark for this tool, built on the problem definitions
of [MetalBench](https://github.com/Lazarus-931/MetalBench) (MIT, vendored at
the commit in `problems/PIN`): 116 MLX modules in three sets, single ops
(`common`), fused pairs (`standard`) and small full models (`full`), each with
registered input shapes and tolerances. Only the problems are taken. Their
timing harness and leaderboard are not: they divide GPU-only kernel time by
wall-clock MLX time, which reports dispatch overhead as speedup.

Each problem becomes one ordinary optimization job. The bridge generates a model
file and manifest from the problem's shapes and tolerances. The harness checks
correctness and compares the whole workload with the selected baseline.

Pass `--baseline compiled` to require improvements over `mx.compile`. The CLI
currently defaults to `plain`, which requires improvements over eager execution.
Use the same baseline, shapes, judge settings and budgets when comparing runs.

The scoreboard shows measured speedups against both compiled and eager execution.
An unchanged model scores 1.0 against its selected baseline. The kernel column
shows isolated region results, which are not necessarily installed model gains.
`fast_p` is the fraction of all problems strictly faster than p; a failed job
counts as not faster. Compare runs against the same execution mode; a gain over
eager execution does not establish a gain over compiled execution.

By default a kernel is installed when it beats plain eager MLX, the bar KernelBench and the
other kernel benchmarks use; `--baseline compiled` holds it to the model under `mx.compile`
instead, the harder bar. Eager runs write `<chip>.eager[.tag].json`, so the two never mix.
Either way both columns are measured: the finished model is timed against the other
baseline at the end of every job.

```bash
uv run python metalbench/fetch.py                       # once, or to re-vendor
uv run python metalbench/run.py --set standard --judge claude-cli
uv run python metalbench/run.py --only abs,rms_norm_linear --budget-per-region 4 --budget-total 8
```

Results accumulate per chip in `runs/metalbench/results/`. Problems the tool cannot cover
(no scaffold for an op, nothing shipped) score 1.0 and count against it,
which is the point of a benchmark.

## Output layout

All new output lives under `runs/metalbench/`, ignored by Git. `results/` contains
scoreboards; `jobs/<timestamp>/` contains generated manifests, per-problem logs,
reports and artifacts. Each invocation uses a fresh jobs directory, including
`--rerun`, so previous kernels and logs are retained. Scoreboard rows include
`work_dir`. Use `--output-dir /path/outside/the/repo` to store everything elsewhere.
Existing legacy output is left in place because it can contain absolute paths.
