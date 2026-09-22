# metal-autotune

Optimize GPU kernels inside an MLX model on Apple Silicon.

Give the tool a model and the input shapes you care about. An AI proposes Metal
kernels; an independent harness checks their outputs and measures the whole
workload. Confirmed improvements become a portable code bundle you can apply to
a compatible model. Finding no confirmed improvement is a valid result.

## Results: up to 2.34× over compiled MLX, 3.12× over eager MLX

**19 of 35 MetalBench standard workloads improved beyond `mx.compile` on an
Apple M4.** Of all 35, 17 exceeded 1.1× and 8 exceeded 1.25× against compiled MLX.

Across all 35 workloads, geometric mean speedup was **1.16× over compiled MLX** and **1.46× over eager MLX**.

| Workload | vs compiled MLX | vs eager MLX |
|---|---:|---:|
| Group normalization | **2.34×** | **3.12×** |
| Instance normalization | **2.23×** | **3.06×** |
| Cross-entropy loss | **1.54×** | **1.83×** |
| Scaled dot-product | **1.47×** | **1.47×** |
| SwiGLU | **1.39×** | **1.45×** |

Both columns compare whole-workload wall-clock time against the compiled
optimized version, using warmed repeated calls. The eager comparison includes
compilation's own gains. The 16 workloads without a shipped kernel score 1.00×
against compiled MLX. These are synthetic workloads, not full-model speedups.

MLX 0.32.2; `claude-fable-5-1` at low effort; up to 30 attempts per problem.
[All 35 results, raw timings, kernels, and reproduction instructions](metalbench/published/2026-09-19-m4/README.md).

## Quickstart

Use an Apple Silicon Mac. The project uses Python 3.12 and MLX 0.32.2;
`uv` installs the pinned Python dependencies for you.

1. Install [Apple's Command Line Tools](https://developer.apple.com/library/archive/technotes/tn2339/_index.html)
   if needed: run `xcode-select --install` in Terminal and finish the installer.
   They provide the C++ compiler used by the graph extension.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
   [Claude Code](https://code.claude.com/docs/en/setup). Run `claude` once,
   complete sign-in, then exit its session. You need an account with Claude Code
   access. Existing Codex and Gemini CLI logins are also supported.
3. Get the repo and run the small example:

```sh
git clone https://github.com/shivamg05/metal-autotune.git
cd metal-autotune
uv sync --locked
uv run autotune run examples/tiny_mlp.yaml --judge claude-cli
```

Already have the checkout? Run the last two commands from its root directory.
Keep this terminal open until the job finishes. The tool checks the judge
connection before loading the model and builds its graph extension on first use.

This example uses a small randomly initialized model and downloads no weights.
Judge calls may consume your provider's subscription allowance or API credits.
Keep other GPU work quiet. Search and validation take longer than a single model
call, and an attempt budget is not a time limit.

The CLI prints its output paths. New jobs default to a unique folder under
`runs/`; use `--work-dir /your/output/path` to choose another location.

- `report.json`: measurements, attempts, failures, and final outcome.
- `run.jsonl` and `candidates.log`: progress and per-candidate results.
- `artifact/`: produced only when a final improvement is confirmed and export passes.

The final summary tells you whether an artifact was verified, no improvement
was confirmed, or the job failed. An isolated kernel speedup is not a shipped
model speedup. Use [the artifact guide](docs/artifacts.md) after a verified win.

## Run through an AI agent

For progress updates without reading logs yourself, open this repo in a coding
agent after completing setup and give it this prompt:

```text
Follow RUNNING.md to run metal-autotune on manifest.yaml with --judge claude-cli
using a fresh work directory under runs/ labeled work-<date-time-est>.
Do not look at other runs' work directories or artifacts. Along the way, give
concise updates at milestones: initial measurements, region changes, accepted
improvements, errors, and final validation. Report the final result and artifact
path, or explain why nothing shipped.
```

Replace `manifest.yaml` with your chosen workload; use `examples/tiny_mlp.yaml`
for the download-free example. The supervising agent launches and monitors the
run. `--judge claude-cli` selects the separate AI that proposes kernel changes;
it does not have to be the same provider as your supervising agent.

## Model examples

The repo includes [model definitions and workload manifests](models/README.md)
for Qwen, Llama, Mamba, RecurrentGemma, and Whisper, plus a randomly initialized
FLUX transformer. The Stable Diffusion integration requires extra setup. These show how to target language,
audio, and image models. Model weights are downloaded when needed, not committed.

For example, optimize an MLX-LM request with a 128-token Qwen3-4B prompt and
one generated token, against the compiled baseline:

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

MIT.
