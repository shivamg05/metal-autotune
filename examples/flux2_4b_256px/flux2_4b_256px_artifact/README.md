# Optimized `models/flux2_4b.py`

This folder makes the model in `model/models/flux2_4b.py` faster on Apple Silicon. It
comes from one metal-autotune job: parts of the model now run custom Metal GPU
kernels, and the job checked that the outputs still match the original. The
folder is self-contained, so you don't need the optimizer to use it.

## Result

- `denoise`: **1.105x faster** over 10 consecutive steps (11,759.3 ms untouched, 10,638.9 ms patched), confirmed. For one step: 1,180.6 ms untouched, 1,086.9 ms patched (1.086x), confirmed above the timing noise.
- `long_prompt`: **1.102x faster** over 10 consecutive steps (17,909.1 ms untouched, 16,252.6 ms patched), confirmed. For one step: 1,804.9 ms untouched, 1,664.3 ms patched (1.085x), confirmed above the timing noise.

Measured on an Apple M4 against the same model run under `mx.compile`, with both versions timed together at the end of the job. Other machines, weights and input sizes can move these numbers.

## Quick start

Run these from inside this folder. You need an Apple Silicon Mac and Python 3.12.13.

```sh
pip install -r requirements.txt   # pins mlx 0.32.2
python validate.py                # patched outputs match the original on the saved inputs
python benchmark.py               # re-time original vs patched, the way the job did
```

## Use it in your code

Put this folder next to your code and import it by its folder name. If
you rename the folder, change the import to match.

Build and load your model the way you normally do, patch it, and use the
returned model:

```python
from flux2_4b_256px_artifact import apply

model = build()                              # your original builder
model.load_weights("my_weights.safetensors")  # if you load weights
model = apply(model)                         # keep the returned model
outputs = model(*inputs)
```

To build the model from the bundle's own copy of the source instead:

```python
from flux2_4b_256px_artifact import load
model = load().inference_model
```

**What it works with.** The kernels were made for this model's exact
operations, layer layout, shapes, dtypes and quantization. Different weight
values with the same structure (for example, a fine-tune) are fine. A
different architecture or quantization needs a new optimization run. Inputs
whose shapes the job didn't measure run the original code: correct, just not
faster. Check speed and correctness on your own weights and inputs before
relying on the result.

**Weights are not included.** `build()` loads or initializes weights exactly
as the original does: a Hub model uses your cache or downloads, a local
checkpoint must be reachable, and random initialization stays random.
`bundle.json` records the checkpoints the job saw (`weight_sources`).

## Reference

### What was replaced

Each row is a set of modules (by attribute path from the model root) and the
kernels that now run inside them. A kernel runs only on the input shapes and
dtypes the job recorded; anything else falls back to the original code.

| modules | count | kernels |
|---|---|---|
| `double_blocks.{0..4}.attn` | 5 | `r49ca8c4c87ffe25b_tg256_denoise` |
| `single_blocks.{0..19}.attn` | 20 | `r52a9dd684fcb7af4_h1`, `r49ca8c4c87ffe25b_tg256_denoise`, `r61a68b6b5360d461_tile64` |
| `double_blocks.{0..4}.ff.linear_in`, `double_blocks.{0..4}.ff_context.linear_in` | 10 | `rf4ff944e9e62e804_h1_tile64` |

### Loading options

`load()` runs the bundled `build()` and applies the patch. Calling the returned
object (`loaded(*inputs)`) reproduces the job's benchmark interface;
`loaded.inference_model` is the patched model for normal use.
`load(patched=False)` gives the untouched model; for a side-by-side comparison
with identical weights use `original = load(patched=False, measurement_baseline=True)`
and `patched = load(share_weights_with=original.model)`.
`measurement_baseline=True` restores the job's original compiled scopes for
timing; without it, `patched=False` leaves the model untouched for correctness
checks. Because the job measured against the compiled model, `load()` runs the forward pass under `mx.compile`; pass `compile=False` for the plain model.

### How it was checked

`validate.py` runs every saved workload through the patched and the original
model and applies the job's correctness rule. Edits that keep floating-point
evaluation unchanged must match bit for bit. Edits that change it must satisfy
`abs(patched - original) <= atol + rtol * abs(original)` with the tolerances in
`bundle.json`. Shapes, dtypes and non-floating values must match exactly, and
non-finite values must sit in the same places. The whole patched model is
compared at once, so several replacements share one allowance.
`python validate.py --sequences` also checks consecutive steps and cache state.
`validate.py` exits nonzero on a mismatch.

`benchmark.py` runs that check first, then times whole runs of 10
consecutive steps for both models, alternating their order across 4 pairs
with cooling in between. It exits nonzero unless the patched model is faster by
more than the measurement's own noise. `--steps` and `--pairs` override the
job's settings.

### Files

- `manifest.yaml`: the manifest the job ran, as written (when the job had one);
  its paths point to where the job ran.
- `model/`: the model's source, copied unchanged.
- `workloads/`: the inputs the job measured on, one file per workload:
  - `denoise`: (1, 256, 128), (1, 256, 7680), (1,)
  - `long_prompt`: (1, 256, 128), (1, 512, 7680), (1,)
  - `denoise@I=64`: (1, 64, 128), (1, 256, 7680), (1,)
  - `denoise@I=1024`: (1, 1024, 128), (1, 256, 7680), (1,)
  - `long_prompt@I=64`: (1, 64, 128), (1, 512, 7680), (1,)
  - `long_prompt@I=1024`: (1, 1024, 128), (1, 512, 7680), (1,)
- `kernels/`: one `.metal` body and `.launch.json` per kernel. A staged kernel
  keeps its shader bodies in `<id>.stages/0.metal`, `1.metal`, and so on.
- `patch/wrappers.py` and `swap_table.json`: which module each kernel installs
  into and how.
- `runtime/`: the small package that loads the kernels and installs them. It
  checks MLX, Python and platform compatibility before loading.
- `load.py`, `apply.py`, `validate.py`, `benchmark.py`: the entry points above.
- `bundle.json`: entry file, baseline, versions, workloads, tolerances and patches.
- `report.json`: the job's full record: every region, attempt and measurement.
- `requirements.txt`: the pinned packages. `buffers/` is reserved and empty.

### Changing it

A kernel is `kernels/<id>.metal` (the body; MLX generates the signature from
the input and output names) plus `kernels/<id>.launch.json` (the launch
arithmetic). Edit either, then run `validate.py` and `benchmark.py` again, and
don't use a kernel that no longer matches the original's outputs.
`patch/wrappers.py` is ordinary Python. To change the model itself, edit
`model/` and run the optimizer again: the patch is tied to the exact operations
it recorded.
