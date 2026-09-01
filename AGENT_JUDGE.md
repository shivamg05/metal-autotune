# Running a job without an API key

The judge is stateless per call: every call carries the full region state, and the
reply is one strict JSON object. That means any model with a keyboard can be the
judge. Two transports make that real; neither touches `ANTHROPIC_API_KEY`.

## Option 1: the local Claude Code login does the judging

```bash
uv run autotune run manifest.yaml --judge claude-cli
```

Each judge call spawns one `claude -p` process on this machine's Claude Code
login, with every tool disabled (`--tools ""`), no settings, no slash commands,
and an empty working directory, and with the harness's judge system prompt
replacing the default. The judge therefore sees only what the harness sends,
which keeps the boundary law intact: metadata, never tensors, weights, or
tolerance values. `--model` picks the judge model as usual.

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
   `system` holds your role and the exact response schema for this call;
   `messages` holds the region state, and on a re-ask also your rejected reply
   plus the rejection reason. Answer the last user message.
2. Write `NNNN.response.json` into the same directory. Its entire content must
   be one JSON object matching the schema in `system`. No prose, no code fences.
   The harness accepts the file the moment it parses as JSON, so chunked writes
   are fine as long as the pauses stay under 10 seconds; a file that stops
   changing without ever parsing is taken as-is after that grace window and
   rejected through the normal path.
3. A response outside the schema gets exactly one re-ask (the next request
   file), then that call counts as a failed hypothesis. The files stay behind
   as the audit trail.

Be honest about what this mode guarantees. The harness still sends metadata
only, but unlike the other transports nothing can stop an agent that lives in
this repo from reading the work dir's tensors, measurements, or the manifest's
tolerances before answering; option 2 trades that mechanical guarantee for
convenience and relies on you honoring the boundary. So honor it: judge from
the request files alone, never from the repo or work dir; the harness decides
correctness and speed, never you. You plan a hypothesis queue in English and
write Metal only for the front ready item, one small edit of a named parent at
a time. Return `"kernel": null` to yield when out of ideas. `DESIGN_SPEC.html`
section on the judge and `autotuner/judge/schema.py` define every field. The
request files remain afterward, so a reviewer can audit what the judge was
told and what it answered.

Both transports share the API client's exchange loop (`autotuner/judge/client.py`),
so validation, the single re-ask, and JudgeBabble behave identically everywhere.
