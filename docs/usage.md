# Running an optimization

Start with the [quickstart](../README.md). This page covers custom models, manifests, and troubleshooting. Commands run from the repository root.

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
uv run autotune run manifest.yaml --judge claude-cli --work-dir runs/work1
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
| `agent` | you answer the AI's requests yourself through files in `<work-dir>/judge_io/`; [the judge protocol](judge-protocol.md) explains how |

Each transport receives the same JSON contract. The Codex preset reads its
final-response file so progress messages cannot corrupt the reply. Omitting
`--model` uses the selected provider's default. See [the judge protocol](judge-protocol.md) for details.

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

For an AI operator, set up the event-driven monitoring in [the operator guide](operating.md) immediately
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
See [the judge protocol](judge-protocol.md) for the reference and lookup formats.
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


## Results and deployment

See [Using an artifact](artifacts.md).
