# Using an artifact

When a run succeeds, it prints `verified artifact` and a folder path. That
folder is the artifact: a self-contained code bundle that makes your model use
the faster kernels. You don't need this optimizer installed to use it.

(The `checkpoints/` folder is not the artifact. Checkpoints are saved during the
search in case the run dies, and haven't passed the final checks.)

## Apply it to your model

Copy the folder into your project as `artifact/` and install its pinned
dependencies:

```sh
python -m pip install -r artifact/requirements.txt
```

Then load your model the way you normally do and patch it. For an MLX-LM model
whose `build()` returns the model directly:

```python
from mlx_lm import load, generate
from artifact import apply

model, tokenizer = load("/path/to/my-checkpoint")  # or a compatible Hub model ID
model = apply(model)
text = generate(model, tokenizer, prompt="Explain gravity simply.")
```

For a custom model, build and load it the same way your original `build()`
does, then call `apply(model)`. **Always use the returned model.** If your
`build()` wraps the model in something, recreate that wrapper too.

**What has to match.** The kernels were built for one exact model structure, so
the operations, module layout, weight shapes, dtypes and quantization all have
to match the model that was optimized:

- **Different weight values, same everything else** (e.g. a fine-tune that
  keeps the same layers and quantization): fine as is.
- **Different architecture or quantization** (e.g. 4-bit → 8-bit): run the
  optimizer again.

Either way, check correctness and speed on the weights and workload you'll
actually ship.

**Or let the bundle build the model.** `from artifact import load`, then use
`load().inference_model`. This runs the bundled copy of your `build()` and
gives you the patched model. It loads whatever weights `build()` picks; there's
no argument to point it at a different checkpoint. Each artifact's own
`README.md` shows both routes, plus a custom-model example.

### Decode (`context`) runs

If the run optimized a decode step (a workload with `context`), `load()` rebuilds
exactly the step that was measured. It's what the benchmark uses. For real
generation, where your code owns the cache and it keeps growing, use
`load().inference_model`.

As the cache grows past the positions the run tested:

- **Replaced parts that touch the cache** (e.g. an attention block that reads
  or writes it) fall back to the original code at any position or cache layout
  that wasn't tested.
- **Replaced parts that don't** (e.g. weight projections) keep using their
  kernels as the cache grows.

The final multi-step measurement already includes those fallbacks.

## Weights and other files

**Weights are never copied into the bundle automatically**, and that includes
the recovery checkpoints. The bundled `build()` loads weights exactly as your
original did: a pretrained checkpoint comes from the usual Hugging Face cache,
download or local path, and random weights stay random.

So whatever `build()` needs has to exist on the machine where you run the
bundle: checkpoints, local files, environment settings, extra libraries.

- **Want a local data file included?** Add `ARTIFACT_FILES = [...]` to your
  model file, as a plain list of relative paths. Your model's local Python
  imports are copied automatically; other files are only bundled if they're
  listed there.
- **Checkpoint paths and adapter options** that `build()` works out at runtime
  stay as they are. Export doesn't rewrite them.

**Code dependencies.** Before loading the model, the tool checks that
everything your model file imports can be packaged. Anything outside the
standard setup must be either an installed package or source code inside your
model project.

**Checkpoint versions.** During a run, the tool records which MLX-LM checkpoint
snapshot it loaded and uses that same snapshot for every comparison copy. The
bundle notes the source and revision for reference, but it doesn't force that
revision or point at the original machine's cache. If future builds must get
exactly that revision, pin it in your model file.

## Check it yourself

The bundle carries the exact inputs the run measured, so you can re-check it
on your own machine. From inside the artifact folder:

```sh
python validate.py     # patched vs original outputs on the saved inputs, the job's own rule
python benchmark.py    # repeat the saved final measurement; exits 1 unless a win is confirmed
```

- **`validate.py`** compares the patched model with the bundled original, using
  the run's own rule: exact match, or within the manifest's `rtol`/`atol`.
  It doesn't need any fp32 reference files.
- **`benchmark.py`** reruns the final measurement the way it was done: same
  execution mode (including compiled parts of the model) and, by default, the
  same number of repetitions the run actually used. That can be more than the
  manifest asked for, because short direct-call workloads get extra repetitions.
  Passing `--steps` changes that count, which makes it a different experiment.
  Older bundles that didn't record a count use the manifest's.

Timing noise, or a different machine, can still move the number.

Keep the pinned dependencies in `requirements.txt`.

## How the bundle is checked before you get it

Export builds the bundle in a temporary folder next to the destination and only
moves it into place once it passes. If writing or checking fails, your accepted
checkpoints and any earlier artifact are left untouched. If the final move
fails, the previous artifact is restored.

The check runs in a fresh process:

- It builds the original and the patched model with the **same freshly loaded
  weights**. (Weights are shared before any decode cache is filled.) That way
  randomly initialized models compare fairly, without freezing their weights
  or comparing two unrelated random models.
- It checks every workload and every sweep size in the manifest, through both
  `apply()` and the bundle's own `load()`, including the cache state for
  decode runs.
- It runs **offline**, using your existing Hugging Face cache. If something the
  model needs is missing, it reports that instead of downloading it mid-export.

If any check fails, nothing is published.

**Where it goes.** By default, `<work-dir>/artifact/`. You can pick another
destination, but it has to be new, and it can't overlap the run's logs,
kernels or checkpoints. The tool rejects a bad path before any GPU work starts.

## What's in the bundle

- Your model's source code, copied unchanged.
- The exact inputs the run measured. For decode runs, this includes the tokens
  that filled the cache, so `load()` rebuilds the same step.
- The kernels and the generated code that plugs them into your model.
- A pinned `requirements.txt`.
- `validate.py` and `benchmark.py`, so the bundle can check and re-time itself.
- A `README.md` written for that run. It states the measured result in plain
  words and lists which part of the model got which kernel.

What each number in `report.json` means is covered in the
[usage guide](usage.md#whats-in-reportjson).
