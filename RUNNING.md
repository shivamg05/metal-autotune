# Running an optimization job

This file is for whoever operates a run, human or AI agent. It assumes you
know nothing else about this project, and it is complete: every step, every
case, and how to report what happened. If you are an agent running this for a
person, section 7 is as binding as the steps.

## What this tool does

It watches your model run, finds spots where custom GPU code could beat the
library's, and writes a correct starting version of each spot's code itself.
Then it asks an AI (the "judge") to improve that code one small edit at a
time. Every suggestion is compiled, checked for identical outputs, and timed.
Only proven wins are installed. The result is a folder (the "artifact") that
makes a freshly loaded copy of your model faster, with none of this machinery
attached.

Expect a run on an 8B model to take one to three hours: about ten minutes of
setup and measurement, then minutes per improvement attempt. The AI proposes;
the tool verifies; most attempts fail. That is the design working, not a
problem.

## Setup

You need an Apple Silicon Mac and `uv`. Nothing else is configured by hand:
the tool measures this machine's own speed limits at the start of every job.
If it cannot measure (no Metal GPU), it stops with an error instead of
pretending.

```bash
uv sync
```

## 1. Write the model file

One `.py` file with a `build()` function at module level. It takes no
arguments and returns your model, ready to call:

```python
def build():
    from mlx_lm import load
    model, _ = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    return model
```

Two rules, both checked before the job starts, both reported with the fix if
you break them:

- The file must import cleanly from any working directory. Resolve paths from
  `__file__`, never from the current directory.
- Slow work like loading weights goes inside `build()`, not at import time.

`build()` is called more than once per job to get untouched copies of the
model, so it must return the same model every time.

## 2. Write the manifest

The manifest is the job order: which model, which input shapes matter to you,
how much searching to pay for.

```yaml
model: llama8b.py
workloads:
  - name: prefill
    inputs:
      - shape: [1, 512]
        dtype: int32
        low: 0          # integer inputs sample from [low, high)
        high: 128000    # keep token ids inside the vocabulary
budget: {per_region: 8, total: 40}
```

That is a complete manifest. Optional settings:

- Write a letter instead of a number (`shape: [1, L]`) for a size that varies
  in real use, and `primary: {L: 512}` picks the size that is traced and
  timed (default: the largest entry of `sweep`, which is 4096 unless you set
  it). Checking installed code at the other `sweep` sizes is not wired into a
  job yet, so a result is proven at the primary size only.
- `budget` caps improvement attempts per spot and for the whole job (defaults
  25 and 250). Each attempt costs minutes, so this is the run-length dial.
- `tolerances: {rtol: ..., atol: ...}` changes how exactly outputs must
  match. Leave it out unless you know why you need it.

Precision and quantization are not settings: the model is optimized exactly
as `build()` hands it over.

## 3. Start the run

```bash
uv run autotune run manifest.yaml --judge claude-cli --work-dir work1
```

Three rules before you press enter:

- Use a fresh `--work-dir` every run. The job refuses a used one so two runs'
  records can never mix.
- On a fanless Mac, start cool. The GPU slows itself several-fold when hot
  and recovers after about twenty minutes of idle; a hot start makes
  everything slow and small wins invisible.
- Run it in the background. You will watch its log, not its terminal.

`--judge` picks where the AI suggestions come from:

| option | what it means |
|---|---|
| `claude-cli` | uses this machine's Claude Code login; no API key needed |
| `api` (default) | uses the Anthropic SDK; needs `ANTHROPIC_API_KEY`; `--model` picks the model |
| `agent` | you answer the AI's requests yourself through files in `<work-dir>/judge_io/`; AGENT_JUDGE.md explains how |

## 4. Watch it

The job writes one JSON line per event to `<work-dir>/run.jsonl`, one plain
line per attempt to `<work-dir>/candidates.log` (time, spot, the idea tried,
its verdict), every question to the AI and its answer to
`<work-dir>/judge.jsonl`, and every kernel it checked to `<work-dir>/kernels/`.
The stages arrive in this order:

| stage | log lines you see | typical time | what is happening |
|---|---|---|---|
| load and map | `model`, `trace`, `regions` | 1-2 min | loads the model twice (weights shared, memory does not double) and finds the spots worth trying |
| measure | quiet, then `peaks`, `aa_floor`, `step_clock`, quiet again until `ranked` | 5-10 min | records reference data, measures the machine's limits and its noise, times the untouched model, prices every spot |
| search | `region_open` ... `region_closed`, one block per spot | minutes per attempt | builds starting code, asks the AI for improvements, verifies each one |
| finish | `step_clock`, then the summary on stdout | 1-2 min | re-times the model and writes the artifact |

The measure stage is silent by design; `<work-dir>/session.jsonl` keeps
ticking while it works, so check there before suspecting a hang. Background
load on the machine is fine: every speed comparison runs both versions back
to back, so noise hits both sides equally and can hide small wins but never
create false ones.

## 5. When things go differently

| situation | what to do |
|---|---|
| the job refuses to start | read the message; it names the fix (bad manifest key, model file rule broken, used work dir) |
| the job stops with `job_refused` before searching | the GPU is busy with another process or reading far below normal, usually heat; nothing can be measured honestly; wait for it to go quiet and cool, then start a fresh run |
| the log shows `env_warning` | the machine's noise is lopsided, or the model's weights could not be shared between its copies; the run continues and its results stay valid, but small wins may go unnoticed |
| a spot is skipped (`region_skip`, `scaffold_failed`) | normal; the log line names the reason; report it plainly and move on |
| the AI's replies keep getting discarded (`babble`) | one or two is normal noise; every call failing means a real bug: capture the log and report to the maintainer |
| the job crashes or hangs | a bug in the tool; capture `run.jsonl` and console output for the maintainer; do not edit the tool and rerun |
| it finishes with nothing installed | a real answer, not a failure: nothing beat the library while keeping outputs identical, and the model is unchanged |

## 6. Read the result and use the artifact

The job prints how many spots got a proven speedup and the model's time per
step before and after. `<work-dir>/report.json` has the full account: every
spot, why work on it ended, and every attempt with its verdict.

The artifact folder (default `artifact/`) is self-contained and needs nothing
from this repo:

```python
from artifact.apply import apply
model = apply(build())   # a freshly loaded model, now patched
```

Ship it, commit it, or copy it to another machine of the same chip
generation. The code it installs is exactly the code that was measured.

## 7. How to talk to the person

The log speaks this project's internal language. The person you report to
does not, and must never need to. Before sending anything, reread it as
someone with no context on this work: every sentence must carry value that is
easy to take in.

- Lead with the outcome. The first sentence answers "what happened"; detail
  follows only if it changes what the person does next.
- Plain words first. If an internal term is unavoidable, define it in the
  same sentence, the first time it appears.
- Never paste raw log lines or event names at the person. Translate, using
  the table below.
- Give numbers with a comparison ("47 ms per token, close to the 44 ms
  physical limit"), never bare.
- Write whole sentences. No fragments, arrow chains, or invented
  abbreviations.
- When the person must decide something, state the decision in plain words,
  the options in plain words, and which one you would pick.

| the log says | tell the person |
|---|---|
| `trace`, `regions`, `ranked` | "mapped the model and found N spots worth trying to speed up" |
| `region_open` | "working on spot k of N", plus what that code does in the model |
| `scaffold_ok` | "built a correct starting version of this spot's code" |
| `scaffold_failed` (watchdog) | "our starting version ran too slow to be worth improving, so this spot was skipped" |
| `scaffold_failed` (smoke) | "our starting version produced wrong numbers; the safety checks caught it and the spot was skipped" |
| `region_skip` (no scaffold) | "the tool does not yet know how to write starting code for this operation, so this spot was skipped" |
| `judge ... babble` | "the AI's reply was not in a usable format, so that attempt was discarded" |
| `verdict` failed | "an improvement attempt failed the correctness or speed checks and was discarded" |
| `verdict` correct_slower | "an attempt was correct but not faster; it will not be installed but later edits can build on it" |
| `shipped` | "found a real speedup: outputs identical, measurably faster, now installed" |
| `e2e_failed`, rolled back | "the speedup did not hold up in the whole model, so it was removed; the model is unchanged" |
| `region_closed` | "finished with this spot", plus the reason in plain words |
| `job_refused` | "the machine cannot measure honestly right now (busy or overheated), so the run stopped before starting; try again once it is idle and cool" |
| `env_warning` | "the machine is noisier than usual, so a small speedup might go unnoticed; the results are still valid" |

A check-in while it runs is one or two sentences: which stage, what that
means, roughly when something changes. The final report is three parts: what
got faster and by how much (or the honest zero and what it means), what was
skipped and why in plain words, and anything that needs the person's
decision.
