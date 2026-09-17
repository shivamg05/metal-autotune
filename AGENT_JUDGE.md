# Running a job without an API key

The judge proposes the next kernel change. The harness checks it, measures it,
and decides whether the whole model got faster. Each call contains a complete
briefing and expects one JSON reply. These two ways to run the judge use your
existing agent login, without `ANTHROPIC_API_KEY`.

`DESIGN_SPEC.html` is the authority for the tool's behavior.

## Option 1: an agent CLI does the judging

```bash
uv run autotune run manifest.yaml --judge codex
```

Use `--judge claude-cli` or `--judge gemini` to select another agent. Add
`--model <model-id>` to choose its model. Without that flag, Claude uses the
harness's Claude default; Codex and Gemini choose their own defaults. Codex
ignores user configuration for this judge process, so pass `--model` explicitly
if you want the model you normally select in that configuration.

Every call starts a fresh process in a temporary directory outside the repo.
The Codex preset reads the briefing from stdin, permits a non-Git directory,
disables local instructions and shell, plugin, memory, and other access tools,
and reads only the final-response file. Progress text on stdout cannot become
the kernel reply. The Claude preset disables tools, settings, and slash
commands and supplies the harness's system prompt directly. Gemini receives
the briefing as a prompt argument; its own configuration controls tool access.
CLI calls time out after 10 minutes.

Other headless agents can use `--judge-cmd '<command>'`. In that command,
`{system}` receives the system prompt, `{prompt}` receives the messages, and
`{response}` receives the path to a fresh reply file. Without `{system}`, the
system prompt is included in the briefing. Without `{prompt}`, the briefing
goes on stdin. Without `{response}`, stdout must contain only the JSON reply.
Custom commands control their own tool and file access. An empty working
directory alone does not prevent an agent from reading other files.

## Option 2: you, the agent reading this, are the judge

Run the job in the background, then answer its mail:

```bash
uv run autotune run manifest.yaml --judge agent --work-dir autotune_work
```

The harness writes numbered requests into `autotune_work/judge_io/` and blocks
(30 minutes per call, then the job dies) until you answer each one. The mailbox
must start empty; the job refuses a mailbox holding a previous run's files,
because stale responses would silently answer new requests.

Protocol, per request:

1. A file `NNNN.request.json` appears. It is self-contained:
   `{"seq": N, "respond_in": "NNNN.response.json", "system": ..., "messages": [...]}`.
   `system` holds your role, the rules, and the exact response schema for this
   call. `messages` holds one user message whose content is a JSON object with
   `region_state` (the whole briefing: the region, its kernels, the queue, the
   verdicts so far) and, after the first call, `verdict` (what happened to the
   last kernel). On a re-ask the list also holds your rejected reply and the
   rejection reason. Source lookups also append your request and the harness's
   source reply. Answer the last user message; earlier messages stay available.
2. Write `NNNN.response.json` into the same directory. Its entire content must
   be one JSON object matching the schema in `system`. No prose, no code fences.
   Write to a temporary file and rename it to the response filename when done.
   The harness accepts the response as soon as it parses as JSON. An incomplete
   file that stops changing for 10 seconds is rejected through the normal path.
3. A response outside the schema gets exactly one re-ask (the next request
   file), then that call counts as a failed hypothesis. The files stay behind,
   and `<work-dir>/judge.jsonl` keeps every exchange in order.

Judge from request files alone. The briefing contains shapes, dtypes, operations,
kernel sources, measurements, and previous verdicts. It excludes tensors,
weights, activations, and numeric acceptance settings. An agent working in this
repo can still read those elsewhere, so this mode relies on honoring that
boundary.

Each request is complete on its own. Start with the target, head/shipped kernels,
costs and latest verdict, then use the history and lessons. Large identical code
or measurement details may appear once in `shared_context`;
`{"context_ref": "context_1"}` means the exact value under that name.

Code over 8,000 characters appears as `{"source_id": "source_1"}`. Its
`source_catalog` entry gives its size and exact excerpts near names called by
the kernel, or the beginning if no match is found. These are navigation hints,
not complete functions or a dependency graph. To inspect more, reply with only:

```json
{"read_source": [{"id": "source_1", "find": "helper_name"}]}
```

Or request an exact slice:

```json
{"read_source": [{"id": "source_1", "start": 12000, "length": 8000}]}
```

Opening excerpts share a 12,000-character budget across all source versions,
prioritizing head, the latest candidate and shipped. A catalog entry without
an opening excerpt remains fully searchable and readable.

Offsets count zero-based characters, not bytes. `start` defaults to zero.
Reads allow 1-8,000 characters; literal searches allow 1-200 characters and
return up to 20 matches. Use `next_start` to continue reading or searching.
Batch 1-4 requests in one reply. Up to eight lookup rounds are allowed before
submitting a proposal; valid reads do not spend optimization attempts or run
the GPU. Malformed lookup requests use the ordinary single re-ask. Lookups
cannot include queue mutations or a kernel. All replies so far are carried
forward, even when the CLI starts a fresh process. Each new proposal starts
with a fresh catalog. This works through the same JSON exchange for every
provider, with no new CLI tool permissions or filesystem access.

Failed-attempt history and worked failure examples remain. Return actual Metal
strings, never reference objects or incomplete excerpts as code. Omit `header`
to inherit the named parent's full unchanged header, or supply a complete
replacement. Read missing helpers before relying on their behavior.

Plan hypotheses in English and write Metal only for the front ready item after
your queue edits. A region opens wide first: while `widening.left` is above 0,
each kernel is an opener written against the scaffold under a kind no earlier
opener used (`directions` lists what can pay under the region's bound; your own
direction is welcome), unless it repairs a kernel that failed. After the
openers, each reply makes one focused change to a named parent. Explain
the mechanism and expected effect in `hypothesis`, then use the next verdict
to decide what to try. The suggested moves are examples; a different algorithm
or layout is allowed if it follows the laws in the briefing. Keep lessons tied
to measured results. A faster isolated kernel is only a candidate: `shipped`
means it also passed installation and whole-model checks. A yield
(`"kernel": null`) is refused while the region's budget lasts: the harness asks
again, and after one free re-ask every reply with nothing to evaluate costs an
attempt, so spend the budget on kernels. `DESIGN_SPEC.html`
describes the judge's job; `autotuner/judge/schema.py` defines every reply
field and `autotuner/judge/prompts.py` every briefing field. The
request files remain afterward, so a reviewer can audit what the judge was
told and what it answered.

Use the kernel proposal's optional `target_workload` to name the workload this
edit aims to improve, using a workload label from the briefing. The harness
checks correctness across all applicable workloads and sweep inputs, but tests
speed on the nominated workload first. Shipping requires a resolved whole-model
win on that workload, repeated in an independent confirmation, and no resolved
regression on any other declared workload. An unchanged workload need not win;
an unresolved measurement is not proof of zero slowdown. Results stay separate
by workload, so extra targets do not dilute a useful gain into an average. The
final bundle is measured against the untouched model and requires at least one
resolved workload win and no resolved regressions.

Both transports share the API client's exchange loop (`autotuner/judge/client.py`),
so validation, the single re-ask, and JudgeBabble behave identically everywhere.

## Candidates with several GPU stages

A proposal can contain one Metal body or a complete `stages` list. Use the
response schema in the request for the exact fields. For example, a split-K
matrix multiply can write partial sums to `tmp0` in stage 0, then reduce `tmp0`
to the region's `out0` in stage 1. This is one candidate and one attempt.

Each stage lists its input and output buffer names, body, launch, output shapes
and output dtypes. Its code and shape expressions use **local** slots: with
`inputs: ["in2", "tmp0"]`, local `in0` refers to region input 2 and local `in1`
refers to `tmp0`. Its first output is local `out0`, whatever name the output
list assigns it. Outer `output_shapes` still refer to the original region inputs.

Return the entire sequence when editing a staged parent. Omitted stage headers
inherit the proposal's shared header, which defaults to the parent's top-level
header. Stage-specific parent headers must be supplied again. Inputs and earlier
results cannot be overwritten; every intermediate must feed a region output.
Templates use local input dtype references. All final output dtypes stay fixed.
These rules also apply when replacing an existing custom Metal kernel.

The harness evaluates the complete sequence, including intermediate allocations.
All original outputs, including model state, still go through the numeric gates.
Extra stages provide algorithmic freedom; their overhead still has to earn its
place in the whole-model measurement. No model changes or extra provider tools
are needed.

## Following a run

The terminal reports selection and pricing progress. `run.jsonl` records phase
changes and measured verdicts; `candidates.log` gives one line per kernel
attempt. `judge.jsonl` records each request before waiting, followed by a reply
or transport error. Its `event` field is `request`, `reply`, or `error`. A final
request with no reply or error means the process was still waiting when logging
stopped; it is not evidence that a kernel was evaluated.

If you are also operating the run for a person, follow the required milestone
monitoring and reporting rules in [RUNNING.md, section 7](RUNNING.md#7-how-to-talk-to-the-person).
That is the single reporting contract: announce every accepted model win,
region switch and error, plus the baseline measurement and final result.
Keep status narration outside the judge's JSON reply. An isolated judge call
only returns its proposal; the operating agent owns watching and check-ins.

A worker GPU evaluation has a five-second deadline, including first-use JIT.
Cooling has a separate allowance within the overall worker budget. A timeout
stops the whole job; it is not a request for another kernel repair. Preserve
the candidate and logs for investigation. A worker process shares the desktop
GPU, so killing it does not guarantee that its submitted GPU work was cancelled.
