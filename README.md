# metal-autotune

Optimize GPU kernels inside an MLX model on Apple Silicon.

Give the tool a model and the input shapes you care about. An AI proposes Metal
kernels; an independent harness checks their outputs and measures the whole
workload. Confirmed improvements become a portable code bundle you can apply to
a compatible model. Finding no confirmed improvement is a valid result.

## Results: up to 2.34× faster than compiled MLX

**19 of 35 MetalBench standard workloads improved on an Apple M4.**
17 of 35 exceeded 1.1×, and 8 exceeded 1.25×, against `mx.compile`.

| Workload | Speedup vs compiled MLX |
|---|---:|
| Group normalization | **2.34×** |
| Instance normalization | **2.23×** |
| Cross-entropy loss | **1.53×** |
| Scaled dot-product | **1.47×** |
| SwiGLU | **1.39×** |

These are whole-workload wall-clock speedups: **compiled original time ÷ compiled
optimized time**, using warmed repeated calls. The other 16 workloads shipped no
kernel and score 1.00×. This is a synthetic benchmark suite, not a claim that an
entire language model or every input shape gets the same speedup.

The run used MLX 0.32.2, `claude-fable-5-1` at low effort, and up to 30 attempts
per problem. [All 35 results, raw timings, workload definitions and final kernels](metalbench/published/2026-09-19-m4/README.md)
are included, along with the reproduction command and follow-up verification notes.

## Quickstart

You need an Apple Silicon Mac, Python 3.12, [uv](https://docs.astral.sh/uv/),
and a supported judge provider. For the example below, install the Claude CLI
and complete its login first. Codex and Gemini CLIs are also supported.

From this checkout:

```sh
uv sync
uv run autotune run examples/tiny_mlp.yaml --judge claude-cli
```

This example uses a small randomly initialized model and downloads no weights.
Judge calls may consume your provider's subscription allowance or API credits.
Keep other GPU work quiet. Search and validation take longer than a single model
call, and an attempt budget is not a time limit.

The CLI prints its output paths. New jobs default to a unique folder under
`runs/`; use `--work-dir /your/output/path` to choose another location.

- `report.json`: measurements, attempts, failures, and final outcome.
- `run.jsonl` and `candidates.log`: progress and per-candidate results.
- `artifact/`: produced only when a final improvement is confirmed and export passes.

An isolated kernel speedup is not a shipped model speedup. Read the final
confirmation and workload timings in the report.

## Model examples

The repo includes [model definitions and workload manifests](models/README.md)
for Qwen, Llama, Mamba, RecurrentGemma, Whisper, and Stable Diffusion, plus a
randomly initialized FLUX transformer. These show how to target language,
audio, and image models. Model weights are downloaded when needed, not committed.

For example, optimize a 128-token Qwen3-4B prompt:

```sh
uv run autotune run models/workloads/qwen3_4b_prefill_128.yaml --judge claude-cli
```

Check the example index for dependencies and workload limits. The small MLP in
the quickstart is the simplest starting point and needs no download.

## Use your own model

Provide a Python file with `build()` returning a callable MLX model, and a YAML
manifest describing inputs, correctness tolerances, and attempt budgets.
The [usage guide](docs/usage.md) explains the contract. Existing model adapters
and their dependencies are listed in [models/README.md](models/README.md).

```sh
uv run autotune run path/to/manifest.yaml --judge codex --model YOUR_MODEL_ID
```

To stop searching while still finishing validation and packaging:

```sh
uv run autotune finish --work-dir runs/YOUR_RUN
```

## Use the optimized model

Copy the resulting artifact into your application and install its
`requirements.txt`. Load a compatible model with your normal weight-loading API,
then use the return value of `artifact.apply(model)`. The bundle does not copy
weights by default. Shapes outside verified coverage use the original computation.
See [artifact usage and compatibility](docs/artifacts.md) and the README generated
inside each artifact.

## Benchmark and develop

[MetalBench](metalbench/README.md) runs a suite of small workloads:

```sh
uv run python metalbench/run.py --set standard --baseline compiled \
  --budget-per-region 10 --budget-total 30 --judge claude-cli --tag first-run
```

- [RUNNING.md](RUNNING.md): starting point for an agent operating a run.
- [Architecture and invariants](docs/architecture.md): how the harness works.
- [Judge protocol](docs/judge-protocol.md): custom providers and file-based judging.
- [Limitations](docs/limitations.md): what results and coverage do not guarantee.

Source lives in `autotuner/` and `autotuner_runtime/`. Tests stay in `tests/`;
optional measurement diagnostics live in `tools/`. Generated runs are ignored by Git.

## License

MIT. Vendored MetalBench problems retain their [upstream MIT notice](metalbench/problems/LICENSE).
Pull requests are welcome; see the [development notes](docs/architecture.md#development).
