# Running an optimization

This is the full path from "I have a model" to "I have a faster model". If you
haven't installed anything yet, do the [quickstart](../README.md#quickstart)
first. Run every command here from the repo root. Want a ready-made target
instead of your own? Pick one from the [model catalog](../models/README.md).

## 1. Provide a model

Write a Python file with a `build()` function that returns your MLX model,
ready to call:

```python
def build():
    from mlx_lm import load
    model, _ = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    return model
```

The tool calls `build()` several times, to get separate "original" and
"optimized" copies it can compare. That's where these rules come from:

- **Load inside `build()`, not at import time.** That includes downloads and
  weight loading.
- **Return a new model every call.** Don't keep one model in a global and hand
  it back each time. (The tool shares weights between the copies itself.)
- **Resolve local file paths relative to `__file__`**, so the file still works
  when it's run from another directory.
- **Using random weights?** Seed them inside `build()` so separate runs get the
  same model.

You don't have to change your model code. The tool finds the MLX operations,
and any custom Metal kernels you already use, on its own; no hooks needed. It
never changes your model's precision or quantization. Parts of the model it
can't handle are skipped and listed in the report.

## 2. Write a manifest

The manifest is a short YAML file that says which inputs to make faster and how
many attempts to allow. [Write a manifest](manifest.md) has examples to copy,
including time to first token, a single decode step and a plain model call, plus
every available field. Save one as `my_run.yaml` in the repo root and change its
model path and inputs. Paths inside the manifest are relative to the manifest file.

## 3. Run it

```sh
uv run autotune run my_run.yaml --judge claude-cli
```

What to expect:

- **It takes a while.** Depending on the model and budget, a run can take hours.
  Long pauses are normal: the tool rests the GPU between timings so heat
  doesn't skew them, and it waits on the AI for each new kernel.
- **Keep the terminal open and other GPU work quiet.**
- **One job at a time.** The tool refuses to start a second one.
- **Output goes to a new folder under `runs/`**, and the tool prints the path.
  If you pick your own with `--work-dir`, it has to be empty.
- With a CLI judge or `--judge-cmd`, the tool sends one small test request
  before loading the model to make sure the AI is reachable. That doesn't count
  as an attempt.

**Choosing the AI (the "judge").** The judge is the AI that writes kernels.

| `--judge` | What you need |
|---|---|
| `claude-cli`, `codex`, `gemini` | That CLI installed and signed in. `--model` picks its model. `--judge-effort` sets effort for `claude-cli`. |
| `api` (the default if you leave it out) | `ANTHROPIC_API_KEY` set. `--model` picks the Anthropic model. |
| `--judge-cmd "<command>"` | Any headless command you like. See the [judge protocol](judge-protocol.md). |
| `agent` | Requests are written to files and someone (or some agent) answers them by hand. See the [judge protocol](judge-protocol.md). |

To have a coding agent run and watch the job for you, point it at
[RUNNING.md](../RUNNING.md). The [operator guide](operating.md) lists the
updates it should send you.

**Stopping early.** To stop new attempts but still get the final checks and the
packaged result, run this from another terminal:

```sh
uv run autotune finish --work-dir runs/YOUR_RUN
```

Whatever is in progress finishes first. Don't just kill the process: you'd lose
the final checks and be left with only the recovery checkpoints. There's no
resume command.

## 4. Read the result

A run ends in one of three ways:

- **Verified artifact.** You have a faster model. See [Using an artifact](artifacts.md).
- **No confirmed improvement.** The run worked, but nothing it found beat the
  baseline by a margin it could confirm.
- **Failure.** Something broke. See [troubleshooting](#troubleshooting).

**How to read the speedup.** Speedup = original time ÷ optimized time, for that
workload against that baseline. So `1.25x` means the work takes 20% less time,
not 25% less.

For MLX-LM generation runs, you'll also see generated tokens/sec before and
after. That's output tokens divided by the *whole* request time, prompt
processing included, so it isn't pure decode speed. Direct model-call runs
report time per call instead.

**Only the final result counts.** During the search you'll see kernels that
are fast on their own, and changes that get accepted along the way. Neither is
the result. The result is the final whole-workload comparison.

What's in the output folder:

- `report.json`: every measured workload, the baseline, all attempts, the final
  checks and the outcome.
- `run.jsonl`: progress events, one per line. `candidates.log`: a readable
  record of each attempt.
- `checkpoints/`: changes accepted during the search, saved in case the run
  dies. They haven't passed the final checks.
- `artifact/`: the bundle you actually use. It's only written after the final
  comparison confirms a win and the bundle passes validation.

### What's in `report.json`

At the end, the tool prints how many regions got a confirmed speedup, and two
before/after timings: one step of the model, and a longer sequence of
consecutive steps. Both are measured against the baseline in your manifest
(compiled unless you changed it). The final comparison is what decides whether
anything ships. It measures the task your manifest describes, which may be
narrower than your whole application.

`report.json` has the full story: every region and why work on it stopped,
every attempt and its verdict, and the step timings for both plain and compiled
MLX. The fields you'll most likely look at:

| Field | What it tells you |
| --- | --- |
| `step_ms[workload].win_confirmed` | Whether the final measurement confirmed a speedup from an installed kernel. **This is the one to trust.** A `speedup` ratio on its own can be noise, especially when nothing was installed. |
| `step_ms[workload].speedup_vs_plain` | With the compiled baseline: the finished model against plain, uncompiled MLX. This includes what compiling alone buys you. |
| `step_ms[workload].speedup_vs_compiled` | With the plain baseline: plain MLX plus the new kernels, against the untouched model under `mx.compile`. Below 1 means compiling alone would have been faster. |
| `plain_win_confirmed` (and its siblings) | Whether that cross-baseline comparison was confirmed. It's measured in pairs at the end of the job. |
| `final.passed`, `reason`, `outcome` | If outputs were right but the win couldn't be confirmed, the job still ends normally: `final.passed` is false with a `reason`, `outcome` is `unconfirmed`, all measurements are kept, and no artifact is written. Wrong outputs end the job with an error instead. |
| `step_ms[workload].min_win_ms` | The smallest per-step saving a region's change needed to count as a win there: 1% of the step, capped at 30 µs. |
| `step_ms[workload].steps_per_sample` | How many back-to-back steps went into each timed sample. More than 1 means a single step was too short to bring the GPU up to full speed on its own. Every whole-model number is still reported per step. |
| `floor_ms` (on each attempt) | A physical floor timed right next to that kernel: an estimate of how fast the hardware could do that work. It's used for ranking, not a hard limit. |

The final timing comes from the same comparison as the final correctness check.

For timing details and what each log field means, see the
[measurement reference](measurement.md).

## Troubleshooting

| What happened | What to do |
|---|---|
| Judge missing or login expired | Follow the printed message, sign in again, and start a fresh run. |
| Compiler missing or extension build failed | Install Apple's Command Line Tools (see the [quickstart](../README.md#quickstart)). If it still fails, keep the compiler error; you'll need it. |
| A region is unsupported | Nothing to do. The tool skips it, keeps searching the others, and notes it in the report. |
| No confirmed improvement | Not an error: nothing passed both the correctness check and the final timing within the budget. It also doesn't prove there's no speedup to find. The timing may just have been too noisy to call. |
| Judge stops responding mid-run | After three requests in a row fail to get through (connection or provider errors, not bad kernels), the search stops and keeps any accepted checkpoints. Fix the provider problem before running again. |
| Crash, worker failure or GPU timeout | Keep the console output, `report.json` and `run.jsonl`, and report which stage failed with its traceback. Don't immediately retry a kernel that timed out on the GPU. |

Background load and heat can hide small speedups. The tool accounts for timing
noise, but it can't promise a win on every model. See [limitations](limitations.md).
