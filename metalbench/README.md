# MetalBench for autotune

A KernelBench-style benchmark for this tool, built on the problem definitions
of [MetalBench](https://github.com/Lazarus-931/MetalBench) (MIT, vendored at
the commit in `problems/PIN`): 116 MLX modules in three sets, single ops
(`common`), fused pairs (`standard`) and small full models (`full`), each with
registered input shapes and tolerances. Only the problems are taken. Their
timing harness and leaderboard are not: they divide GPU-only kernel time by
wall-clock MLX time, which reports dispatch overhead as speedup.

Each problem becomes one autotune job (`bridge.py`): the model file wraps the
vendored `Model`, the manifest names its shapes, tolerances and budget, and the
harness measures as it always does, paired and interleaved, with the compiled
model as the baseline and the ladder deciding correctness.

Scores (`run.py`) are the confirmed whole-step speedup per problem, 1.0 when
nothing shipped, against two baselines side by side: the compiled model, the
honest bar, and the eager model, the bar the other benchmarks quote. `fast_p`
is the fraction of problems strictly faster than p. Read the two columns
together: where they coincide the win is a kernel's, where the eager column is
large and the compiled one small the win was the compiler's.

```bash
uv run python metalbench/fetch.py                       # once, or to re-vendor
uv run python metalbench/run.py --set standard --judge claude-cli
uv run python metalbench/run.py --only abs,rms_norm_linear --budget-per-region 4 --budget-total 8
```

Results accumulate per chip in `results/`. Problems the tool cannot cover
(no scaffold for an op, nothing shipped) score 1.0 and count against it,
which is the point of a benchmark.
