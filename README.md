# metal-autotune

Make an MLX model faster on Apple Silicon without changing its answers.

You give the tool a model and the inputs you care about. An AI writes custom
Metal GPU code (kernels) for parts of the model. A separate harness tests every
one, and keeps it only if the outputs still match the original (bit for bit,
or within a tolerance you control when the math is reordered) and the **whole
workload** gets measurably faster. Whatever survives is packaged as a code bundle you
apply to your model.

Sometimes nothing survives. That's a real answer, not a crash: no candidate
beat the baseline by a margin the tool could confirm within its attempt budget.

**Does it work?** On MetalBench, a set of 35 small synthetic GPU workloads, one
run on an Apple M4 made 19 of them faster than MLX's own compiler
(`mx.compile`). The best was 2.34× faster, and the average (geometric mean) across all
35 was 1.16×, counting the ones it couldn't speed up as 1.00×. That's what one
benchmark run found, not a ceiling. See [MetalBench results](#metalbench-results)
for the full breakdown.

## Quickstart

You need an Apple Silicon Mac. `uv` installs the pinned Python 3.12 and
MLX 0.32.2 setup for you.

1. **Install [Apple's Command Line Tools](https://developer.apple.com/library/archive/technotes/tn2339/_index.html)**
   if you don't have them: run `xcode-select --install` and finish the
   installer. The tool needs their C++ compiler to build a small extension.
2. **Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
   [Claude Code](https://code.claude.com/docs/en/setup).** Run `claude` once,
   sign in, and exit. Claude Code is the AI that writes the kernels, so your
   account needs Claude Code access. A signed-in Codex or Gemini CLI works too.
3. **Clone the repo and run the small example:**

```sh
git clone https://github.com/shivamg05/metal-autotune.git
cd metal-autotune
uv sync --locked
uv run autotune run examples/tiny_mlp.yaml --judge claude-cli
```

If you already have the repo, just run the last two commands from its root.

This example is a tiny model with random weights, so nothing downloads. A few
things to know while it runs:

- **Leave the terminal open** until it finishes. Expect it to take a while:
  each attempt gets checked and timed, and the budget counts attempts, not
  minutes.
- **Keep other GPU work quiet.** Background load makes the timings noisy.
- **Kernel requests use your AI provider's allowance** (subscription or API
  credits).
- Before it loads the model, the tool checks that it can reach the AI. It also
  builds its extension the first time you run it.

**What you get.** Results go to a new folder under `runs/`, and the tool prints
its path. Use `--work-dir /your/path` to pick a different folder.

- `report.json`: every measurement, attempt and failure, plus the final outcome.
- `run.jsonl` and `candidates.log`: progress and a result for each attempt.
- `artifact/`: the optimized code bundle. It only appears when the final check
  confirms a speedup and the bundle passes its own validation.

The final summary tells you which of three things happened: a verified
artifact (`job complete: verified artifact ...`), no confirmed improvement, or
a failure. A kernel that
is fast on its own doesn't count; only a confirmed whole-model speedup ships.
If you got an artifact, see [the artifact guide](docs/artifacts.md).

## Run through an AI agent

Don't want to watch logs? Once setup is done, open this repo in a coding agent
(Claude Code, Codex, …) and paste this. The agent starts the run, watches it,
and messages you when something happens:

```text
Follow RUNNING.md to run metal-autotune on manifest.yaml with --judge claude-cli
using a fresh work directory under runs/ labeled work-<date-time-est>.
Do not look at other runs' work directories or artifacts. Along the way, give
concise updates at milestones: initial measurements, region changes, accepted
improvements, errors, and final validation. Report the final result and artifact
path, or explain why nothing shipped.
```

Swap `manifest.yaml` for your own manifest, or use `examples/tiny_mlp.yaml`
for the download-free example.

There are two AIs here, and they can come from different providers. The agent
you're chatting with runs and watches the job. The one named by `--judge`
(`claude-cli` here) is the one that writes kernels.

## Model examples

The [model catalog](models/README.md) has ready-made targets for language,
audio and image models:

- **Language:** Qwen3, Qwen3.5, Llama 3, Mamba, RecurrentGemma
- **Audio:** the Whisper encoder
- **Image:** a FLUX transformer with random weights

It also has newer targets, including LFM2.5, Qwen3.5-9B, Qwen3.8-27B and Muse Glimmer 30B.
These haven't been through a full optimization run yet. Stable Diffusion is
there too, but needs extra setup. Weights download the first time you use a
model and are never committed to the repo.

For example, this makes Qwen3-4B faster at reading a 128-token prompt and
producing one token, measured against compiled MLX:

```sh
uv run autotune run models/workloads/qwen3_4b_prefill_128.yaml --judge claude-cli
```

The catalog lists each model's dependencies and limits. If you just want to try
the tool, the tiny example in the quickstart is still the easiest start.

## Use your own model

You need two files:

1. **A Python file** with a `build()` function that returns your MLX model.
   The [usage guide](docs/usage.md#1-provide-a-model) has the rules, and
   [models/](models/README.md) has working examples to copy.
2. **A manifest**, a short YAML file saying which inputs to speed up, how many
   attempts to allow and, if you want, how strict the correctness check is.
   [Write a manifest](docs/manifest.md) has examples you can copy.

Then run it. Any supported AI can write the kernels. Here Codex does, with
`--model` picking which Codex model:

```sh
uv run autotune run path/to/manifest.yaml --judge codex --model YOUR_MODEL_ID
```

Found enough and want to wrap up early? This stops new attempts but still runs
the final checks and packages the result:

```sh
uv run autotune finish --work-dir runs/YOUR_RUN
```

## Use the optimized model

Copy the `artifact/` folder into your app and install its `requirements.txt`.
Load your model the way you normally do, then patch it:

```python
from artifact import apply
model = apply(model)   # always use the returned model
```

Keep in mind:

- **Weights aren't included by default.** You load them yourself as usual. The model's
  architecture and quantization have to match what was optimized, but the
  weight values can differ.
- **Only tested input sizes get the new kernels.** Anything else quietly runs
  the original code, so it's still correct, just not faster.

More in [the artifact guide](docs/artifacts.md) and the README inside each
artifact.

## MetalBench results

[MetalBench](https://github.com/Lazarus-931/MetalBench) is a Metal adaptation
of KernelBench. We ran the 35 workloads in its standard set on an Apple M4.
They're small synthetic problems, like a norm followed by a matmul, not full
models. The numbers are what one run found on this benchmark, not a ceiling:
other models and workloads can gain more, or less.

- **19 of 35 got faster than `mx.compile`.** 17 of them beat 1.1×, and 8 beat 1.25×.
- **Geometric mean across all 35: 1.16× vs compiled MLX, 1.46× vs eager MLX.**
  The 16 workloads with no win count as 1.00×.

| Workload | vs compiled MLX | vs eager MLX |
|---|---:|---:|
| Group normalization | **2.34×** | **3.12×** |
| Instance normalization | **2.23×** | **3.06×** |
| Cross-entropy loss | **1.54×** | **1.83×** |
| Scaled dot-product | **1.47×** | **1.47×** |
| SwiGLU | **1.39×** | **1.45×** |

How to read the columns:

- **vs compiled MLX** is the harder bar. `mx.compile` is MLX's own optimizer,
  so a win here comes on top of what MLX already does for you.
- **vs eager MLX** compares against plain, uncompiled MLX, so it includes the
  gain from compiling as well.

Both columns measure the same optimized model, timing the whole workload with
warmed-up, repeated calls. Only the baseline changes.

Setup: MLX 0.32.2; judge `claude-fable-5-1` at low effort; up to 30 attempts per problem.
[All 35 results, raw timings, kernels, and reproduction instructions](metalbench/published/2026-09-19-m4/README.md).

To run the same suite yourself ([MetalBench guide](metalbench/README.md)):

```sh
uv run python metalbench/run.py --set standard --baseline compiled \
  --budget-per-region 10 --budget-total 30 --judge claude-cli --tag first-run
```

## Further reading

- [Limitations](docs/limitations.md): what a result does and doesn't promise.
  Worth reading before you quote a number.
- [RUNNING.md](RUNNING.md): where an agent running a job for you should start.
- [Architecture](docs/architecture.md): how the harness works and the rules it never breaks.
- [Judge protocol](docs/judge-protocol.md): how to plug in another AI provider,
  or answer kernel requests by hand.

Code lives in `autotuner/` and `autotuner_runtime/`, tests in `tests/`, and
optional measurement scripts in `tools/`. Run output under `runs/` is ignored
by Git.

## License

MIT.
