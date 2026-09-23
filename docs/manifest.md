# Write a manifest

A manifest is a small YAML file that tells the tool three things: **which
model**, **what inputs you want to get faster**, and **how many attempts it
gets**. One manifest = one optimization run.

The quickest way in is to copy the example below, save it as `my_run.yaml` in
the repo root, and change the model and prompt length. Everything else can stay
as is.

Jump to: [decode](#measure-one-decode-step) ·
[other models](#measure-a-model-call) ·
[several input sizes](#optimize-several-input-sizes) ·
[field reference](#field-reference)

## Start here: a language model

This one asks: *how fast can Llama read a 512-token prompt and produce its
first token?* You don't supply a prompt; the tool makes up random token IDs.

```yaml
model: models/llama8b.py        # Python file that loads the model
baseline: compiled              # beat the model with mx.compile turned on
use_library_inference: true     # time MLX-LM's own generate path
workloads:
  - name: first_token           # just a label for the report
    inputs:
      - shape: [1, 512]         # one prompt, 512 tokens
        dtype: int32            # token IDs are integers
        low: 0
        high: 128000            # random IDs are drawn from 0..127999
budget:
  per_region: 8                 # max attempts on any one part of the model
  total: 24                     # max attempts for the whole run
final_benchmark:
  steps: 1                      # generate one token per request
```

Run it from the repo root:

```sh
uv run autotune run my_run.yaml --judge claude-cli
```

Results land in `runs/`. **That's enough to start a run.** The rest of this
page is for other goals and details.

A few things that trip people up:

- **`model` is relative to the YAML file**, not to where you run the command.
  Pick one from the [model catalog](../models/README.md) or
  [write your own](usage.md#1-provide-a-model).
- **`dtype: int32` is the input type, not the model's precision.** Token IDs
  are always integers. This Llama file loads 4-bit weights; precision and
  quantization live in the model file.
- **Keep `high` at or below the model's vocabulary size** when you switch
  models, or the random IDs will be out of range.
- **A "region" is a chunk of the model** the tool tries to replace with a
  faster GPU kernel. The budget counts attempts, not minutes or tokens.

**What gets timed:** the full request from prompt to output token, including
sampling and any extra work MLX-LM queues up. Loading weights and tokenizing
text are not timed.

Set `steps: 32` to time a whole 32-token request instead. That includes reading
the prompt, so the tokens/sec it reports isn't a pure decode speed. For that,
see the next section.

`use_library_inference: true` only works for complete MLX-LM language models
with one integer input shaped `[T]` or `[1, T]`. If your model doesn't fit, the
tool says so before the search starts. Use a
[direct model call](#measure-a-model-call) instead.

## Measure one decode step

Use this to time how long it takes to process *one new token* after the model
has already read 512. Those 512 tokens live in the **cache**, the model's saved
work from earlier tokens. The tool fills the cache before timing and resets it
between every comparison, so both versions start from the same place.

```yaml
model: models/llama8b.py
baseline: compiled
use_library_inference: false    # call the model directly, no sampling
workloads:
  - name: decode
    context: 512                # tokens already in the cache
    inputs:
      - shape: [1, 1]           # one new token
        dtype: int32
        low: 0
        high: 128000
budget: {per_region: 8, total: 24}
final_benchmark: {steps: 1, pairs: 4, warmup_steps: 3}
```

With `steps` above 1, the final benchmark calls the model that many times in a
row, feeding the same token each time and letting the cache grow. Copying the
cache into place is part of that timing.

Limits for now: a manifest with `context` can have only **one workload with one
integer input**, and the model has to know how to create and update a cache.
The bundled MLX-LM models do.

## Measure a model call

For anything that isn't MLX-LM text generation, set
`use_library_inference: false` and the tool times a plain call to the model
(a "forward pass"). This example uses the small bundled model. It needs no
download and takes 128 rows of 256 numbers:

```yaml
model: examples/tiny_mlp.py
baseline: compiled
use_library_inference: false
workloads:
  - name: forward
    inputs:
      - shape: [128, 256]
        dtype: float32
budget: {per_region: 4, total: 8}
final_benchmark: {steps: 10, pairs: 4, warmup_steps: 3}
```

This times the model call and nothing around it. For an image denoiser, that's
denoiser calls, not the whole image pipeline. For a speech encoder, it's encoder
calls, not audio preprocessing or transcription. For a model with several
inputs, see [the FLUX manifest](../manifest_flux.yaml).

## Describe your inputs

Each item under `inputs` is one positional argument, in order. Three items
means the tool calls `model(a, b, c)`, so shapes and dtypes have to match what
your model expects. What each dimension *means* (batch, sequence length, …) is
up to the model; the tool doesn't interpret it.

The tool makes up the input values. For integer inputs, `low` and `high` set
the range: `low` is included, `high` isn't. They don't apply to float inputs.

## Optimize several input sizes

Add one workload per input case you care about. This one targets a short and a
long prompt:

```yaml
model: models/llama8b.py
baseline: compiled
use_library_inference: true
workloads:
  - name: short_prompt
    inputs: [{shape: [1, 128], dtype: int32, low: 0, high: 128000}]
  - name: long_prompt
    inputs: [{shape: [1, 512], dtype: int32, low: 0, high: 128000}]
budget: {per_region: 8, total: 24}
final_benchmark: {steps: 1, pairs: 4, warmup_steps: 3}
```

How this plays out:

- Each attempt aims at one workload. It's kept only if it's clearly faster there
  **and** not clearly slower on any other.
- The final result has to speed up at least one workload and slow down none,
  after allowing for timing noise. Gains aren't averaged, so a big win on one
  can't hide a loss on another.
- More workloads means more checking and timing per attempt, and they all share
  the one `total` budget.

Workload names are only labels. Calling one `decode` doesn't make it a decode
measurement; `context` and `use_library_inference` do that.

## Optional: check extra sizes for correctness

Skip this unless you want the result checked at sizes you aren't optimizing.
Give a dimension a name, like `B`, then say which size to optimize and which
sizes to test:

```yaml
model: examples/tiny_mlp.py
baseline: compiled
use_library_inference: false
workloads:
  - name: forward
    inputs: [{shape: [B, 256], dtype: float32}]
primary: {B: 128}                 # optimize for batch 128
sweep: {B: [1, 32, 128, 256]}     # also check correctness at these sizes
budget: {per_region: 4, total: 8}
```

The sweep only checks correctness. It doesn't make those sizes faster, and it
doesn't promise the new kernel runs there. At any size where a kernel hasn't
been verified, the model just uses its original code. If you want other sizes
to get faster too, list them as separate workloads.

## Field reference

Only `model` and `workloads` are required. Each workload needs `inputs`, and
each input needs `shape` and `dtype`. Anything you leave out gets its default.
Misspelled or unknown fields are rejected. A dotted name like `budget.total`
means a nested field.

| Field | Default | What it does |
| --- | --- | --- |
| `model` | required | Path to a Python file with a `build()` function that takes no required arguments. Relative to the YAML file. |
| `baseline` | `compiled` | What to beat. `compiled` = the model under `mx.compile`; `plain` = the model as it runs today. See [compilation](#notes). |
| `use_library_inference` | automatic | `true` times MLX-LM generation; `false` times direct model calls. Left out, the tool picks based on what the model supports. |
| `workloads` | required | List of input cases to optimize. At least one. |
| `workloads[].name` | `workload0`, `workload1`, … | Report label. Must be unique and non-empty, with no `@`. |
| `workloads[].context` | none | Tokens already in the cache. `0` = empty cache. Only allowed with one workload and one integer input. |
| `workloads[].inputs` | required | Positional arguments to the model, in order. At least one. |
| `workloads[].inputs[].shape` | required | List of positive sizes or dimension names, e.g. `[1, L]`. |
| `workloads[].inputs[].dtype` | required | Input type; see the list below. |
| `workloads[].inputs[].low` / `high` | `0` / `100` | Range for random integers, `low` included, `high` excluded. `low` must be less than `high`. Integer dtypes only. |
| `primary` | largest sweep size | Size to optimize for, per named dimension. |
| `sweep` | `[1, 13, 50, 4096]` | Sizes to check for correctness, per named dimension. Set this yourself when you use named dimensions. |
| `budget.per_region` | `25` | Max attempts on one region. |
| `budget.total` | `250` | Max attempts for the whole run. |
| `tolerances` | depends on output dtype | `{rtol: …, atol: …}`, both required. See [correctness](#notes). |
| `final_benchmark.steps` | `10` | Library mode: tokens generated per request. Direct calls: model calls in a row per sample. |
| `final_benchmark.pairs` | `4` | Number of original-vs-optimized timing pairs. Even, and at least 4. |
| `final_benchmark.warmup_steps` | `3` | Minimum warmup calls. The tool may warm up longer if timings haven't settled. |

All counts in `budget` and `final_benchmark` must be positive integers.

Input dtypes: `bool`, `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`,
`int32`, `int64`, `float16`, `bfloat16`, `float32`, `complex64`. A dtype being
accepted here doesn't mean every operation has a starter kernel for it.

## Notes

**Compilation.** With `baseline: compiled`, plain model calls run under
`mx.compile`, and library generation compiles the parts of the model it
safely can. Models that keep internal state between calls sometimes can't be
compiled from outside; those fall back to plain, and the report says so and
why. Very short final benchmarks may be repeated more times to get a usable
timing, and the result records the counts actually used.

**Cache when `context` is left out.** Library generation starts with an empty
cache. Direct calls get no cache at all.

**Correctness.** A change that does the exact same math must give the exact same
output bits. A change that reorders floating-point math can round slightly
differently, so it's allowed an error of `atol + rtol * abs(original)`.
Integer outputs always have to match exactly. The defaults:

| Output dtype | `rtol` | `atol` |
| --- | ---: | ---: |
| `float32` | `0.00001` | `0.000001` |
| `float16` | `0.01` | `0.02` |
| `bfloat16` | `0.02` | `0.04` |

Setting `tolerances` overrides these. Both values must be finite, non-negative
and fit in a float32. On top of the error limit, the checker also compares
shape and dtype, catches NaNs and infinities, and where it applies, allows for
how much the original model's own output varies from run to run.

## What doesn't go in the manifest

- **Checkpoint, weights, quantization:** in the model's Python file.
- **Which judge, judge model, output folder:** command-line flags, e.g.
  `--judge claude-cli --model YOUR_JUDGE_MODEL --work-dir runs/my-run`.
- **Hardware, random seed, cooling:** the tool handles these.

For setup, stopping a run cleanly and reading results, head back to the
[usage guide](usage.md).
