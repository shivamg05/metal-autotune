# Running an optimization job

This file is for whoever operates a run, human or AI agent. It assumes you
know nothing else about this project. It covers starting a job, handling its
reported outcomes, and explaining what happened. If you are an AI agent operating this
for a person, read "For an AI agent operating this run" just below before you
do anything; it and section 7 bind as much as the steps.

## What this tool does

It watches your model run, finds spots where custom GPU code could beat the
library's, and writes a correct starting version of each spot's code itself.
Then it asks an AI (the "judge") to improve that code one small edit at a
time. Every suggestion is compiled, checked for correct outputs, and timed.
Only proven wins are installed. The result is a folder (the "artifact") that
makes a freshly loaded copy of your model faster, with none of this machinery
attached.

Preparation time depends on forward-pass cost and the distinct regions
measured. A FLUX.2 run reached its first shortlist in about seven minutes on
an M4 Air, including cooling. Search attempts can still take minutes each.
An exhausted budget with no measured speedup is a valid result. It finishes
with a report and no artifact; packaging is skipped when nothing was accepted.

## For an AI agent operating this run

These rules apply to operating a job. An explicit request to debug or improve
the tool authorizes development and testing as well.
Follow the numbered steps in order and do nothing outside them. These rules
bind as much as the steps.

- **Assume the repo is healthy. Do not run the test suite, any single test,
  the spikes, or timing scripts of your own.** They take minutes and, worse,
  they load the GPU, and a busy GPU throttles the chip this job must measure
  honestly, which hides real wins. Before and during a run, the only thing
  that should touch the GPU is the run itself.
- **The whole flow is the numbered steps.** Confirm the manifest names the
  model and input shapes you mean (section 2), start the run (section 3),
  watch it (section 4), report back (section 7). A ready-made manifest such as
  `manifest.yaml` needs only sections 3, 4, and 7. Do not invent a different
  sequence.
- **Do not edit the tool to make a run work.** If the job refuses to start,
  the message names the fix, which may involve the manifest, model file,
  or judge login/configuration. If the tool itself crashes or hangs, that is a bug:
  capture the logs (section 5) and hand them to the maintainer; never patch it
  and rerun.
- **Files you may create or edit:** a model file under `models/`, a manifest
  (a `*.yaml` file), and your own `--work-dir` (the run creates it; you only
  read what lands there).
- **Files you must never edit:** the tool itself, in `autotuner/` and
  `autotuner_runtime/`; the tests in `tests/` and the experiments in
  `spikes/`; the project's design documents; and this guide.

## Setup

You need an Apple Silicon Mac and `uv`. Nothing else is configured by hand:
the tool measures this machine's own speed limits at the start of every job.
If it cannot measure (no Metal GPU), it stops with an error instead of
pretending.

```bash
uv sync
```

## 1. Write the model file

The harness starts tracing before importing your model and captures custom Metal
kernel definitions during every model build, including later comparison copies.
Your model does not need tracer setup or capture hooks. Captured custom Metal kernels can be searched individually using their original
source as the starting point, and can participate in supported fusions.
Preserving edits must match every output bit-for-bit. Arithmetic-reordering
edits use the manifest tolerance against the untouched original model. Unsupported calls are
reported before search, and signatures outside the optimized workload use the original.

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
model, so it must return the same weights every time. Seed random weights
inside `build()` or load a fixed checkpoint; artifact validation rebuilds it
in a separate process. Return fresh model and layer instances each time.
Reusing a module-global layer would let an installation alter the baseline,
so the tool rejects it before search. Weight arrays may be shared.

The model file also decides what there is to win. The tool never changes a
model's dtypes or quantization, and on a model decoding one token at a time
the weight stream alone can be most of the step. The `step_floor` line in
the log and `coverage.step_floor` in the report say, before any search
starts, an estimated lower limit and the remaining headroom. Compute costs
use each operation's dtype, and never-evaluated work is left out. The chip's
measured peaks are the limit's ceiling; when the step or a priced spot is
measured running faster than those peaks allow, the peaks rise to what was
seen (a `peaks` line says so), so the room is never negative. This estimate
guides search; only measured forward-pass improvements count as wins.

## 2. Write the manifest

The manifest is the job order: which model, which input shapes matter to you,
how much searching to pay for. Workload names cannot contain `@`, which is
reserved for generated shape labels.

```yaml
model: models/llama8b.py
workloads:
  - name: prefill
    inputs:
      - shape: [1, 512]
        dtype: int32
        low: 0          # integer inputs sample from [low, high)
        high: 128000    # keep token ids inside the vocabulary
budget: {per_region: 8, total: 40}
```

That is a complete manifest. Model files live in `models/`; the manifests in
the repo root are ready-made jobs (`manifest.yaml` is the FLUX.2 Klein4B
denoiser with fixed random weights; the two `manifest_llama_*.yaml` files
are Llama 3 8B jobs, a prompt and a decode step over the same model file).
Optional settings:

- `use_library_inference: true` measures the supported model library's normal
  inference routine. Currently this supports complete MLX-LM language models
  with one prompt, shaped `[T]` or `[1, T]`. One measurement processes that
  prompt and finishes generating `final_benchmark.steps` tokens. Loading and
  tokenization are excluded; sampling, cache updates and queued GPU work are
  included. Every candidate, confirmation and final result uses this same task.
  An existing `context` is prepared before timing; each trial starts with an
  independent copy. Workload names remain arbitrary labels.
  `false` measures the model's forward call, including denoisers, encoders and
  custom callables. When omitted, the tool chooses library inference if the
  model and inputs are supported, otherwise forward execution, and records
  its choice before search. Explicit `true` with an unsupported model or input
  fails before search. Encoder-only and denoiser-only files do not imply a full
  transcription or image-generation pipeline.

- Write a letter instead of a number (`shape: [1, L]`) for a size that varies
  in real use. `primary: {L: 512}` picks the size that is traced and timed
  (default: the largest entry of `sweep`). Every kernel is also checked for
  correct outputs at the other sizes in `sweep` (default `[1, 13, 50, 4096]`),
  and at any size other than the primary the installed code hands the work
  back to the model's original code, so the result is correct at every size
  and faster at the primary one.
- `context: 512` on a workload means 512 tokens of conversation are already in
  place when the call runs. That is what a decode step is: `shape: [1, 1]`
  plus `context: 512`. The job builds the model's cache the model's own
  way (`make_cache()`), fills it once with 512 synthetic tokens, and restores
  its full state after every call, so every call is the same step; the model file needs
  nothing for this. A workload with a context is the only workload in its
  manifest and has one input, the token ids. `context: 0` supplies an empty
  cache for first-prompt processing. In forward mode, omitting `context`
  supplies no cache. In library inference mode, the library creates an empty
  cache when no context is given. Names such as `prefill` are labels only.
  Attention, recurrent and sliding-window caches with Python state fields
  use the same snapshot/restore path, including failures partway through a call.
  The timed call includes cache writes, even when they do not affect its
  logits. An exported model can instead use a caller-owned cache that advances
  normally; untested shapes or cache positions fall back to the original code.
- `budget` caps improvement attempts per spot and for the whole job (defaults
  25 and 250). Each attempt costs minutes, so this is the run-length dial. A
  spot keeps going until its share of the budget is spent, whatever the
  verdicts: the AI is asked again when it runs out of ideas, and a reply with
  nothing to try costs an attempt. A spot's first few attempts, up to four,
  are each a fresh design written against the starting code in a different
  direction the tool lists; after those the AI edits its best kernel. A spot
  whose only remaining cost is the kernel launch itself, with the last three
  kernels at that floor and the best one unmoved for two attempts, closes
  early and the log says so.
- `tolerances: {rtol: ..., atol: ...}` sets the allowance for edits that change
  floating-point evaluation: `abs(new - original) <= atol + rtol * abs(original)`.
  `rtol` scales with the original value; `atol` covers values near zero. The same
  pair applies to region, whole-model, consecutive-step and artifact checks.
  Values must be finite, non-negative and fit in float32, which performs the comparison.
  Without it, recorded per-dtype defaults apply. Other edits must match bit for
  bit. The judge cannot change these values, and integer outputs stay exact.
- `final_benchmark: {steps: 10, pairs: 4, warmup_steps: 3}` shapes the last
  timing of the job. With library inference, `steps` is the number of generated
  tokens in every whole-model comparison throughout search, and final timing
  repeats that same task. With forward measurement, it runs whole sequences of `steps` consecutive forward
  passes on the workload inputs, original and patched, each sequence
  uninterrupted, the two alternated in both orders `pairs` times with cooling
  in between. The speedup the job reports is this comparison, and the
  artifact is written only if it confirms a win.
  For `context` workloads, the final sequence advances an isolated KV cache
  from the saved prefix. Both arms receive the same input tokens at each
  step, and correctness covers every output in the sequence. This measures
  controlled decode, including cache growth, rather than text sampling.
  Prefix-copy setup is included equally in both sequence times.
- `baseline: compiled` (the default) means "faster" is measured against the
  model run under `mx.compile`, which is the faster way to run it and so the
  honest bar; `baseline: plain` measures against the model exactly as
  `build()` returns it. A step that keeps state in Python, such as a decode
  step writing its KV cache, cannot be compiled from outside the model, so
  the job uses the plain baseline for it and says so in the log (`baseline`
  line) and the report. Both timings are recorded whenever both can be taken.

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
| `api` (default) | the Anthropic SDK; needs `ANTHROPIC_API_KEY`; `--model` picks the model |
| `claude-cli`, `codex`, `gemini` | that agent's own CLI, run headless in an empty directory, using the login the CLI already has; no API key; `--model` picks the model |
| `--judge-cmd "<command>"` | any other headless agent CLI; the prompt goes on the command's stdin, or in place of a `{system}`/`{prompt}` token in the command |
| `agent` | you answer the AI's requests yourself through files in `<work-dir>/judge_io/`; AGENT_JUDGE.md explains how |

Each transport receives the same JSON contract. The Codex preset reads its
final-response file so progress messages cannot corrupt the reply. Omitting
`--model` uses the selected provider's default. See `AGENT_JUDGE.md` for details.

For CLI judges, the tool first sends one small readiness request through the
same command, model, environment, and response channel used for search. It
allows up to 60 seconds and requires the expected JSON response. This costs
one short judge call, no optimization attempts, and no model GPU work. A
failure stops startup and prints the CLI's diagnostic from both output streams.
If Claude reports expired authentication, sign in with `claude auth login` in
your terminal, then start a fresh run. An operating agent must report that
required action; it must not invent credentials, strip authentication settings,
or keep restarting the job. API and file-mailbox judges do not use this CLI check.

A successful readiness check cannot prevent a later service outage or expired
session. Three consecutive transport failures during search stop the job,
preserving its partial report and accepted checkpoints. They do not consume
optimization attempts or trigger work on further regions. This is an
incomplete run, not a successful search that found no improvements.

## 4. Watch it

For an AI operator, set up the event-driven monitoring in section 7 immediately
after launch. Every accepted win, region switch and error needs an update.

The job writes one JSON line per event to `<work-dir>/run.jsonl`, one plain
line per attempt to `<work-dir>/candidates.log` (time, spot, the idea tried,
its verdict), every question to the AI and its answer to
`<work-dir>/judge.jsonl`, and every kernel it checked to `<work-dir>/kernels/`.
A candidate may contain several ordered GPU stages. Its shader bodies live in
`kernels/<id>.stages/`, with their wiring and launches in `<id>.launch.json`.
It still counts as one attempt and is measured as a complete replacement.
Each judge request records `context_chars`: the character count of its system
text plus all message text, including any malformed-reply retry. This is not
a provider token count. Large supporting code is introduced with focused
excerpts; the judge can request exact slices or literal searches from the
harness's source catalog. `source_rounds` counts those lookups within the
current proposal. They are navigation, not candidate attempts or GPU work;
they do add model calls when used. Identical code and measurement details are
shared within each request, and failed attempts and lessons remain available.
See `AGENT_JUDGE.md` for the reference and lookup formats.
The stages arrive in this order:

| stage | log lines you see | typical time | what is happening |
|---|---|---|---|
| load and map | `model`, `trace`, `regions` | 1-2 min | loads the model twice (weights shared, memory does not double) and finds the spots worth trying |
| measure | `step_clock`, `peaks`, `aa_floor`, `step_floor`, `selection`, `pricing`, `ranked` | depends on step cost | times the model before calibration, then measures distinct regions sharing model samples |
| search | `region_open`, then `region_closed` | minutes per attempt | builds starting code, asks the AI for improvements, verifies each one |
| finish | `step_clock`, then the summary on stdout | 1-2 min | runs final validation, then packages accepted improvements; a no-win run finishes with its report |

Preparation checks that a region has a delivery scope and a buildable
starter. A CPU estimate selects regions whose recorded spans do not overlap.
The others stay queued and become available as selected regions finish.
Repeated regions are grouped only when their required inputs and outputs have
the same roles and order. A final layer that returns fewer live values is a
separate target. Saved input/reference file headers are checked before each
wave is priced or searched; a mismatch stops with the exact file and reason.
Measured ranking uses Amdahl's law: the region's share of the step times the
fraction of its cost that its estimated limit could remove. Both parts of that
limit are clocked beside the spot in the same window: a kernel that only
streams its bytes, and a large plain matmul at its dtype for the arithmetic
(`measured_compute_gflops` in the pricing table).
`price_stability` in the pricing report shows agreement between observations:
closer to 1 means more consistent, while small values mean the ranking is noisy.
It is a diagnostic, not a confidence interval or proof of no headroom.
The report's `pricing` table retains measured regions, largest share first,
including their rejection reasons. `coverage.discovery` distinguishes legal
regions from those the tool can currently build and install. Unsupported
does not mean unprofitable. The roofline is a scheduling estimate, not a
guaranteed upper bound on a real kernel's performance.

`report.json` exists from startup and records failures as well as results.
Its `accepted` list records each completed full-model promotion immediately,
even if the region has not finished. Each acceptance also saves the exact
kernels, wrappers, runtime, and measurement report under
`<work-dir>/checkpoints/accepted-0001/` (then `0002`, etc.). These are recovery
packages: final comparison and fresh-process artifact validation are still
pending, and the checkpoint report says so. A later crash does not erase them.
There is no automatic resume command; retain the checkpoint and logs if the
run stops, and report them to the maintainer.
`session.jsonl` records each cooling pause before sleeping, with its duration,
then records completion. Long pauses also appear on stdout. The default cooldown
is three times the accounted work. `cooling_scheduled` means a completed
comparison or correctness check has returned its verdict and CPU work may use
that cooldown. Candidate workers also return their remaining cooling deadline
to the parent (`cooling_adopted`), so process cleanup, preparation and judge
thinking can use the same interval. The parent waits before the next worker or
model evaluation; `cooling_reused` reports that remainder.
A pending cooldown does not mean the verdict is still being measured.
`ladder_result.result.detail.pacing` records accounted work and time spent
sleeping inside each successful worker phase. Parent waits appear in
`session.jsonl`. These are wall-clock accounting figures, not direct GPU-active
time: compilation and mixed CPU/GPU checks are still charged conservatively.
Warmup entries include their first-call time and total work; `off_clock_work`
records capture/correctness phase time. The cooling ratio, warmup rules, sample
counts and speedup requirements are unchanged.

The judge normally returns its initial plan and first kernel together. Once a
region has an installed kernel, `incumbent_screen` reports a fresh local
comparison against it. A resolved local regression skips whole-model timing;
this screening result is not a measured whole-model regression. Identity wrapper
checks share model passes while comparing each scope separately.

The `trace` row's `never_evaluated` count is work the model builds but never
runs (MLX is lazy; a result nothing returns, keeps, evaluates or reads never
executes), such as the logits of a prompt-processing call the library drops.
It is left out of regions, prices and the step floor, and
`coverage.discovery.never_evaluated_ops` totals it.

How a win gets installed: the `installation` row names the method for each
module it touches. `direct` means the replacement covers the whole module
call and is one guarded kernel call. `graph` means the original module builds
its calculation as usual, the replacement is substituted into that graph, and
the result is compiled once and reused for calls of the same shapes. `replay`
means generated Python re-runs the module's recorded operations with the
replacement spliced in; it is kept for scopes graph substitution cannot
preserve, such as one whose compiled identity is not bitwise the original
or one that reaches state it was not handed. Both graph and replay serve
only the cache positions the job recorded: a job that records position N
optimizes position N. A `graph_fallback` row names why a scope moved to
replay, `delivery_settled` lists the split before the search, and `graph_verified`
records that both the inspected and the compiled calculation contained exactly
the expected substitutions. Every method passes the same correctness and
whole-model timing checks, and the artifact carries whichever was measured.

The baseline is the compiled model, as everywhere else. Under library
inference it is realized as the model with its outermost compilable scopes
compiled and nothing inserted (`delivery_settled` names the scopes); that
model is installed empty as the job's starting state, every kernel composes
into it, and every whole-model number is measured against it. The artifact
carries it. `baseline: plain` in the manifest keeps the plain model as the
baseline instead; kernels are then measured against their scope compiled with
the cuts it carried before (`incumbent_compiled_scopes` on the accepted row).
Each region's report row says its `delivery` and which `library_arm` its
clocks ran (a graph scope's library ops run as one compiled graph, the way the
deployed scope will).

Candidate workers have a separate five-second limit for each GPU evaluation,
including the first compilation and launch. Cooling does not consume that
limit; it still counts toward the overall worker budget. If either deadline
expires, the job stops and records the reason. Workers share the desktop GPU,
so killing one does not prove its submitted GPU work has stopped.

Keep other GPU work quiet. Interleaved measurements reduce drift, but cannot
guarantee that arbitrary background load affects both arms equally. The macOS
GPU utilization counter does not measure remaining throughput; compare it
with measured bandwidth, compute throughput, and timing noise.
The CLI prevents a second CLI job from running at the same time. If the A/A
control finds a significant difference between identical code twice, the
run stops before search because those measurements could create false wins.

## 5. When things go differently

| situation | what to do |
|---|---|
| the job refuses to start | read the message; it names the fix (bad manifest key, model file rule broken, used work dir) |
| the log shows `env_warning` | read the observation: high utilization alone does not establish contention; low throughput or unstable timings can obscure small wins |
| a spot is skipped (`region_skip`, `scaffold_failed`) | normal; the log line names the reason; report it plainly and move on |
| judge readiness fails before loading | follow the printed login/configuration instruction; preserve console output; do not retry unchanged settings |
| the AI's replies keep getting discarded (`babble`) | these are unusable answers, distinct from connection failures; capture the log if persistent |
| the judge becomes unavailable during search | three consecutive transport failures stop the job; preserve the partial report and accepted checkpoints, then resolve the reported connection problem |
| the job crashes or hangs | a bug in the tool; capture `run.jsonl` and console output for the maintainer; do not edit the tool and rerun |
| a worker exits with a Python exception (`WorkerFailed`) | the tool preserves the traceback and stops; report the exception as a worker failure, not a GPU hang |
| a GPU evaluation or worker times out | the whole job stops; preserve the logs and candidate files, and investigate before another run; do not submit a GPU health probe or immediately retry the candidate |
| it finishes with nothing installed | no attempt established a correct, sufficiently certain whole-model improvement within the budget; the model is unchanged; distinguish slow code from inconclusive timing using the recorded verdicts |

## 6. Read the result and use the artifact

The job prints how many spots got a proven speedup, the model's time per
step before and after, and the time for a whole sequence of consecutive
steps before and after, both against the baseline the manifest chose
(compiled unless you said otherwise). The sequence comparison is the one
that decides: it times the model the way it is deployed.
`<work-dir>/report.json` has the full account: every spot, why work on it
ended, every attempt with its verdict (each with `floor_ms`, the physical
floor timed beside that kernel), and both step timings, plain and compiled.
`step_ms[workload].win_confirmed` says whether the final measurement resolved
a speedup for an installed replacement. A nominal `speedup` ratio alone can
be noise, especially when nothing was installed. The final timing record
comes from the same comparison as the final whole-model validation.

Before publishing the artifact, the tool loads it in a fresh process and
checks every declared workload and sweep against the patched model, both
through `apply()` and through the bundle's own `load()`. A failed check never
publishes the staged result. The CLI requires a fresh destination and defaults
to `<work-dir>/artifact/`. It rejects paths that overlap the run's logs,
kernels, or checkpoints before starting GPU work.

The artifact folder is a code bundle that needs no optimizer installation:
the model's source copied unchanged, the input
tensors the job measured on (and, for a workload with a `context`, the tokens
that filled the cache, so `load()` rebuilds the same step), the kernels and
generated wrappers, a pinned `requirements.txt`, and a `README.md` written for
that job that states the measured result in plain words and lists which part
of the model got which kernel. It loads, verifies, and re-times itself:

For normal inference, copy `artifact/` into your project, install its
`requirements.txt`, load your chosen compatible weights, then apply the patch.
For an MLX-LM model whose original `build()` returns the model directly:

```python
from mlx_lm import load, generate
from artifact import apply

model, tokenizer = load("/path/to/my-checkpoint")  # or a compatible Hub model ID
model = apply(model)
text = generate(model, tokenizer, prompt="Explain gravity simply.")
```

For custom models, use the original `build()` and weight-loading API before
`apply(model)`. Always use the return value. The operations, module layout,
weight shapes, dtypes, and quantization must remain compatible with the model
that was optimized. If the original builder adds a wrapper, recreate it too.
Different weight values do not require copying weights into the artifact;
architecture or quantization changes require another optimization run. Verify
correctness and speed on the weights and workload you actually deploy.

Alternatively, `from artifact import load` followed by
`load().inference_model` builds from the bundled source and exposes the patched
model for normal inference. This uses `build()`'s original weight selection;
`load()` has no separate checkpoint-path argument. Each artifact's `README.md`
includes both loading routes and a custom-model example.

For workloads with `context`, `load()` is the repeatable benchmark step. Use
`load().inference_model` for normal inference with an advancing caller-owned
cache. A wrapper that includes cache-dependent work falls back to the
original module at untested positions or cache layouts; stateless projection kernels remain
usable as the cache grows. The final sequence measurement includes those
fallbacks.

`build()` retains its original weight-loading behavior: pretrained checkpoints
come from their usual cache, download or local path, and random initialization
remains random. No weights are automatically copied into final artifacts or
recovery packages. `ARTIFACT_FILES` is an explicit opt-in for local resources
that the author wants included; other external paths/settings must be available
where the model runs. Dynamic checkpoint paths and adapter options remain the
builder's responsibility rather than being rewritten by export.

Before loading, the tool checks that model imports can be packaged. External
dependencies need an installed distribution or source in the model project.
Actual MLX-LM snapshots are captured during loading and held fixed for later
comparison builds within the same run. The bundle records those sources and
revisions as provenance, without redirecting its loader to the original
machine's cache paths. Pin a checkpoint revision in the model source when you
need future builds to select that exact revision.

Search finishes each region in ranked order before opening the next. It spends
the full per-region budget, including starter repairs, unless the total budget
is exhausted, the operator requests finalization, or the starter is unsupported.
A win keeps the same search going with its queue and history intact. Remaining
regions are repriced after an installed improvement while search budget remains.

Export writes into a temporary sibling directory and validates there before
publication. A write or validation failure preserves accepted checkpoints and
any existing artifact; a failed publication restores the previous artifact.
The bundle records whether outputs must match exactly or within the manifest
rtol/atol allowance. Validation always compares against the bundled original;
new runs do not require fp32 reference files.
Fresh-process validation builds original and patched models with identical
fresh weights and checks the recorded workloads and cache state. Weight sharing
happens before prefix caches are filled. This supports random initialization
without freezing weights or comparing unrelated random models. Saved test inputs
remain in the bundle for repeatable validation and benchmarking; model weights
do not. Validation uses the existing Hugging Face cache in offline mode, so it
will report missing external resources instead of downloading during export.

```sh
python validate.py     # patched vs original outputs on the saved inputs, the job's own rule
python benchmark.py    # re-time original vs patched the way the final check did; exits 1 unless the win holds
```

When the manifest's baseline was `compiled`, `load()` returns a callable that
already runs under `mx.compile`, so what you time is what the job timed.
Ship it, commit it, or copy it to another machine of the same chip
generation. The code it installs is exactly the code that was measured; to
change the model itself, edit the source and run the optimizer again.

## 7. How to talk to the person

### Required milestone monitoring

When operating for a person, start a read-only background watcher as soon as
section 3 launches the job. Use the session's existing process/wait facilities;
no installed service or separate optimization process is needed. Watching is
part of running the job, not something to wait for the person to request.

- Follow this run's `run.jsonl` from the beginning, keeping a byte offset or
  line cursor. The file may not exist yet at startup. Read only complete JSON
  lines; leave a partial last line for the next read.
- The watcher must return or notify the operating agent as soon as any trigger
  below appears, or the optimizer process exits. Do not wait for several
  attempts to finish or use a fixed multi-minute reporting delay.
- Handle all unseen milestones in order, then resume watching from the saved
  cursor. Do not repeatedly announce old wins. Related events from the same
  failed attempt may share one update; distinct accepted wins must all be reported.
- Watch process exit independently of the log. A startup failure or killed
  process may never write `job_failed`. On exit, drain remaining complete lines
  and inspect the exit status, console output and `report.json` before reporting.
  Keep watching through export until the optimizer exits, even after a win or
  `artifact_checked` event.
- If the session cannot deliver background notifications, use bounded foreground
  waits that return on a matching event or process exit, then resume them.
  Do not promise automatic check-ins that the session cannot actually deliver.

| trigger in `run.jsonl` | required update to the person |
|---|---|
| `step_clock` with `phase: before` | Name the model and workload from the manifest, input size and cache context where applicable, baseline mode, and measured `median_ms` per call. Include any recorded measurement caveat; do not wait for calibration or pricing to finish. |
| `region_open` | Say that a new spot is starting. Explain its `ops` in plain language, give `copies`, its combined latency share `p` for each workload, and the per-region and total attempt limits from `job`. Use known module locations if available; do not guess the model block from a fingerprint. |
| `shipped` | Always announce the accepted win. Explain the change using its matching hypothesis/candidate record, give the measured whole-model improvement versus the previous installed version, and the correctness rule used. Link its `checkpoint` and say final validation/export are still pending. |
| `scaffold_failed`, `scaffold_fix`, `scaffold_reference_fallback`, `scaffold_model_fallback`, `bind_failed`, `certification_failed`, `rollback_error`, `plan_refused`, `verdict` with a failed/rolled_back outcome, or `judge` with `action: error`/`babble` | Explain the failure or recovery and what the recorded next action is: repair, another candidate, region skip, or job stop. If the next action is not known yet, say so. Report recovery when a later event establishes it. A correct-but-slower or inconclusive timing result is not an internal error. |
| `env_warning`, `memory_warning` | Explain the observed condition and its possible effect on this run. Do not claim that measurements are invalid, or that recovery happened, without evidence. |
| `region_closed` or `region_skip` | State why the spot finished or was skipped, attempts used when recorded, and whether it contributed an accepted improvement. This can share the next region's opening update. |
| `stage` or `search_finish_requested` | Name the phase now starting and what it does. When search ends, say which validation or packaging step remains. An accepted checkpoint is not yet the finished artifact. |
| `sequence_comparison`, `artifact_checked`, `job_failed`, or optimizer process exit | Report the measured consecutive-step result when available. Announce a finished bundle only after successful export and validation; confirm process exit and final report status. For failure, name the stage/reason, preserved checkpoints and missing final checks. A successful no-win run has a report and no artifact. |

Read `run.jsonl` first. Use `report.json` for full checks and timing details,
`candidates.log` or `judge.jsonl` for the matching optimization idea, and
`session.jsonl` to explain a cooling pause. Read the new log entries before
answering a manual status question, then continue watching. Normal slower
attempts and unchanged cooling/judge waits do not require repetitive updates.
Never invent a finish time. If a win's matching idea has not been written yet,
announce the measured win first and add the explanation when available.

### Explain the evidence simply

Keep milestone updates to one to three plain sentences. Lead with what happened,
then the numbers or explanation needed to understand it and what happens next.
Translate internal names instead of pasting log lines. Define an unfamiliar
operation the first time it appears. Exact wording is flexible; the milestones
and required facts above are not.

- Region timings nominate a candidate; they do not establish a model speedup.
  `shipped.timings` contains repeated whole-model measurements. Use
  `model_latency_reduction_pct`, or `100 * (1 - timings[workload].median_ratio)`,
  for the percentage reduction in latency. A speedup ratio of 1.10x means about
  9.1% less time, not 10% less time.
- The first accepted win compares with the original model; later wins compare
  with the previously installed version. Do not add their percentages.
  `model_ratios` gives an estimated cumulative result for each workload. The
  legacy `model_ratio` is populated only for a single-workload job. Neither is
  the final direct comparison; never present the first workload as the whole job.
- An edit must improve its nominated `target_workload` in both the initial
  whole-model comparison and fresh confirmation, with no resolved slowdown on
  any other declared performance workload. Other workloads may be inconclusive.
  Gains are not averaged across shapes. Name the workloads that improved and
  distinguish unresolved measurements from proof that performance is unchanged.
  Final validation requires at least one resolved win and no resolved losses.
- The final consecutive-step comparison measures the original against all edits
  together. Give the recorded number of steps and both total latencies. If you
  quote per-step averages, label them as averages. Keep this distinct from a
  single-forward measurement and from full text/image generation.
- Correctness passing does not automatically mean bit-identical outputs. Read
  the checks to distinguish exact agreement from agreement within tolerance.
  Estimated headroom is a ranking aid, not a guaranteed physical ceiling.
- A starter failure can lead to a repair or fallback. Do not announce a skip
  until `region_skip` or `region_closed` establishes it. A later crash does not
  erase earlier completed model measurements, but can leave final validation
  and packaging unfinished.

The final response must stand on its own: the workload tested, the final measured
result and whether it was confirmed, artifact/report paths, meaningful skips,
and any unresolved failure or action the person needs to take. If monitoring
was interrupted, catch up from the cursor and identify the gap instead of
presenting a delayed update as something that just happened.

### Stop searching and finish the run

When the operator asks to stop spending attempts and benchmark what was found,
run this in a separate terminal while keeping the optimizer process alive:

```sh
uv run autotune finish --work-dir <work-dir>
```

The request is checked between attempts and regions. The current judge call,
candidate evaluation, or preparation phase finishes safely first. No new region
is opened once the request is seen. The job then runs final correctness and
performance checks and exports only a confirmed win. Watch for
`search_finish_requested`, followed by the final timing stages. This command
writes a request for a running job; it does not resume an exited job. Killing
the process or pressing Ctrl-C interrupts it instead of finalizing it.

An inconclusive single-step check now continues to the consecutive-step
benchmark when correctness and the regression veto pass. Export still requires
a statistically confirmed consecutive-step win.

### Custom Metal kernels and fusion

Captured custom kernels can be searched individually or inside regions with
neighboring operations. A mixed region starts from the recorded original
sequence, including its Metal source and launch settings. That starter may use
several calls; the judge proposes a replacement kernel for the complete region.
Only a replacement that passes correctness, installation and whole-model timing
checks can ship. Untested input signatures fall back to the original module.

Every region uses a measured boundary-data probe as a ranking hint. Known
operation arithmetic adds a compute estimate; arbitrary Metal arithmetic remains
unknown. Neither estimate is a guaranteed physical limit or a headroom gate.
