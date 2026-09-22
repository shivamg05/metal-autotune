# Run an optimization

Start with [README.md](README.md) for installation and a runnable example.
Detailed model and manifest options are in [docs/usage.md](docs/usage.md).

If you are an agent operating a run for someone, read
[docs/operating.md](docs/operating.md) before starting. It defines preflight,
event-driven progress updates, failure reporting, and how to finish early
without losing final validation. Follow the requested manifest and judge;
do not inspect other runs unless asked.

```sh
uv run autotune run path/to/manifest.yaml --judge claude-cli
```

The default output is a fresh timestamped folder under `runs/`. You can specify
`--work-dir runs/YOUR_LABEL` or a location outside this checkout. Never reuse a
nonempty directory or run competing GPU benchmarks alongside an optimization.

After completion, report the confirmed whole-workload result and artifact path,
or explain why nothing shipped. See [artifact usage](docs/artifacts.md).
For a benchmark suite, follow [MetalBench's guide](metalbench/README.md).
