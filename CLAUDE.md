# Working in this repository

Read [README.md](README.md) for the product and commands. If operating an
optimization job, follow [RUNNING.md](RUNNING.md). If changing code, read
[docs/architecture.md](docs/architecture.md), including its development notes.

`DESIGN_SPEC.html` is authoritative for product behavior. Do not silently edit
it to match an implementation or replace it with an implementation plan.

Keep changes small, behaviorally correct, and general across models. Verify
measured limitations before adding machinery. Preserve unrelated worktree edits.
Use targeted tests and run GPU tests serially, with no optimization job active.

The judge proposes candidates; the harness decides correctness and performance.
Never weaken correctness, change model precision, or call an isolated kernel win
a shipped improvement. Keep generated output in ignored `runs/`, not source.

Maintain the relevant user docs and tests when behavior changes. Do not recreate
historical audit ledgers, milestone plans, or append-only platform journals.

Explain work simply and precisely: what changed, why, what was tested, and any
remaining limitation. Avoid jargon when a concrete example conveys the point.

Do not add AI co-author trailers to commit messages. Keep ARTICULATE.md local.
