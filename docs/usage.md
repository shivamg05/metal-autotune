# Running an optimization

Start with the [quickstart](../README.md#quickstart) for installation and a small
example. Commands below run from the repository root. For a larger target,
choose a [model example](../models/README.md).

## 1. Provide a model

Write a Python file with a `build()` function that returns a callable MLX model:

```python
def build():
    from mlx_lm import load
    model, _ = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    return model
```

Put downloads and weight loading inside `build()`, not at import time. Resolve
local file paths relative to `__file__`, so the model works from another working
directory. Return fresh model and layer instances each time; do not reuse a
module-global model. The harness builds independent comparison models and shares
their weights. Seeding random initialization inside `build()` also makes separate
runs repeatable.

The tool keeps the model's precision and quantization. It captures supported
MLX operations and custom Metal kernels automatically; no tracing hooks are
needed in your model. Unsupported regions are reported and skipped.

## 2. Define what to measure

A manifest is a YAML file describing the model, its inputs, and the search
budget. Save this example in the repo root as `my_run.yaml`:

```yaml
model: models/llama8b.py
baseline: compiled
use_library_inference: true
workloads:
  - name: first_token
    inputs:
      - shape: [1, 512]
        dtype: int32
        low: 0
        high: 128000
budget: {per_region: 8, total: 24}
final_benchmark: {steps: 1, pairs: 4, warmup_steps: 3}
```

This measures a complete MLX-LM request for one output token from a 512-token
prompt, including sampling and any work the library queues before returning.
It is not a pure prefill-only timer. Loading and tokenization are excluded. Token IDs are
synthetic, sampled from `[low, high)`; keep that range within the vocabulary.
Model paths are relative to the manifest's directory.

Choose the measurement explicitly:

| Setting | What gets timed |
|---|---|
| `use_library_inference: true` | MLX-LM processes the prompt and produces `final_benchmark.steps` tokens, including sampling and cache updates. `steps: 1` targets the first token; `steps: 32` includes generating 32 tokens. |
| `use_library_inference: false` | Calls the model directly with the declared inputs. For example, one encoder call or one denoiser call. This does not include a surrounding transcription or image-generation pipeline. |

Library inference currently supports complete MLX-LM language models with one
input shaped `[T]` or `[1, T]`. Explicit `true` fails preflight when unsupported.
If omitted, the tool selects a mode based on model/input support and records it.
Workload names are labels, not instructions: naming a workload `prefill` or
`decode` does not configure its behavior.

`baseline: compiled` is the default. Stateless forward calls run under
`mx.compile`. For library inference, the harness compiles supported model scopes
while the library handles generation. Stateful forward calls that cannot safely
be compiled use a plain baseline. The report records the actual choice and
reason; requesting compilation does not guarantee every part can be compiled.
`baseline: plain` compares against the model's existing execution instead.

### Cache and decode

In library inference mode, each measurement starts with a fresh empty cache
unless `context` supplies an existing prefix. In forward mode, no `context`
means no cache is supplied.

- `context: 0` supplies an empty cache.
- `context: 512` prepares a cache containing 512 synthetic tokens before timing.
- A controlled single decode step uses `use_library_inference: false`, token
  input shape `[1, 1]`, and `context: 512`.

During candidate comparisons, the cache is restored between trials so both
models start in the same state. For final forward benchmarks with a cache,
`steps` consecutive calls advance independent copies of that state, using the
same input tokens in both arms. With `steps: 1`, the final check measures one
call. Prefix-copy setup is included equally in these sequence timings.
Currently a manifest with `context` must contain only one workload with one
token input. Supported cache APIs and model-specific limits are listed in the
[model examples](../models/README.md).

### Shapes, correctness, and budget

- Add entries under `workloads` to optimize several specific input shapes.
  A candidate names one workload to improve and must show no resolved slowdown
  on the others. Final confirmation requires at least one resolved improvement
  and no resolved regressions; gains are not averaged across workloads.
- Named dimensions such as `shape: [1, L]` use `primary: {L: 512}` for search.
  `sweep: {L: [32, 128, 512]}` adds correctness checks, not extra optimization
  targets. Without an explicit primary, the largest sweep value is used;
  the default sweep is `[1, 13, 50, 4096]`. Untested signatures use the original
  computation. Workload names cannot contain `@`, reserved for sweep labels.
- `budget: {per_region: 8, total: 24}` permits up to eight attempts on each
  region and 24 overall. A region is a replaceable group of operations. Budgets
  count attempts, not minutes or provider tokens. Defaults are 25 and 250.
  The first up to four attempts explore different designs proposed by the AI;
  later attempts can refine, revisit, or combine designs. See the
  [architecture](architecture.md) for the search flow.
- Arithmetic-preserving changes require bit-identical outputs. Reordered
  floating-point arithmetic uses per-dtype tolerances, or your explicit
  `tolerances: {rtol: ..., atol: ...}`. The check is
  `abs(new - original) <= atol + rtol * abs(original)`. Integer outputs remain
  exact. Tolerances must be finite, nonnegative, and representable in float32.
- `final_benchmark` defaults to `{steps: 10, pairs: 4, warmup_steps: 3}`.
  Forward benchmarks repeat complete calls; short stateless workloads may use
  more repetitions to make timing meaningful. Library inference keeps the
  requested generated-token count. Each pair compares original and patched
  execution, with order balanced across pairs. `pairs` must be even and at
  least four. Warmup and cooling are handled by the tool. Actual repetition
  counts are saved with the result and reused by the exported benchmark.

## 3. Run it

```sh
uv run autotune run my_run.yaml --judge claude-cli
```

The command checks judge readiness before loading the model. Keep the terminal
open and other GPU work quiet. Runs can take hours depending on the model and
budget; cooling pauses and judge requests are normal. The tool prints its log
paths and uses a fresh folder under `runs/` by default. An explicit `--work-dir`
must be empty. Only one CLI optimization job may run at a time.

| Judge | Setup |
|---|---|
| `claude-cli`, `codex`, `gemini` | Install and sign in to the selected CLI first. `--model` selects its model. |
| `api` (default if omitted) | Requires `ANTHROPIC_API_KEY`; `--model` selects the Anthropic model. |
| `--judge-cmd "<command>"` | Custom headless command; see the [judge protocol](judge-protocol.md). |
| `agent` | File-based requests answered by an external operator; see the [judge protocol](judge-protocol.md). |

For CLI judges, the readiness request costs one small provider call, not a
search attempt. `--judge-effort` controls Claude CLI effort. To have an agent
run and monitor the job, give it [RUNNING.md](../RUNNING.md); the
[operator guide](operating.md) defines milestone updates.

To stop new attempts while keeping final validation and packaging, use another
terminal:

```sh
uv run autotune finish --work-dir runs/YOUR_RUN
```

The current work finishes first. Killing the process instead may leave only
recovery checkpoints. There is no automatic resume command.

## 4. Read the result

The final summary reports a verified artifact, no confirmed improvement, or a
failure. A speedup is **original time divided by optimized time**, for the stated
workload and baseline. `1.25x` means 20% less execution time. A faster isolated
kernel or a temporary acceptance during search is not the final result.

- `report.json`: measured workloads, baseline, attempts, final checks, and outcome.
- `run.jsonl`: progress events; `candidates.log`: readable per-attempt records.
- `checkpoints/`: accepted work saved during search, still awaiting final checks.
- `artifact/`: the deployable bundle, only after confirmation and export validation.

See [Using an artifact](artifacts.md) to apply a result. For timing details and
internal log fields, see the [measurement reference](measurement.md).

## Troubleshooting

| Situation | Next step |
|---|---|
| Missing judge or expired login | Follow the printed diagnostic, sign in, and start a fresh run. |
| Missing compiler or graph build failure | Install Apple's Command Line Tools; see the [setup steps](../README.md#quickstart). Preserve the compiler diagnostic if it still fails. |
| Unsupported region | Other regions may still be searched. The report records the coverage limit. |
| No confirmed improvement | A valid outcome: no candidate passed correctness and final performance confirmation within the budget. Inconclusive timing is not proof that no improvement exists. |
| Judge becomes unavailable | Three consecutive transport failures stop search and preserve accepted checkpoints. Resolve the provider error before another run. |
| Crash, worker failure, or GPU timeout | Preserve console output, `report.json`, and `run.jsonl`; report the failing stage and traceback. Do not immediately retry a timed-out GPU candidate. |

Background load and temperature can hide small differences. The tool accounts
for uncertainty; it cannot promise a win on every model. See [limitations](limitations.md).
