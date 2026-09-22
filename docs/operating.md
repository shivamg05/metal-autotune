# Operating a run for someone else

## For an AI agent operating this run

These rules apply to operating a job. An explicit request to debug or improve
the tool authorizes development and testing as well.
Follow the numbered steps in [the usage guide](usage.md) in order and do nothing outside them. These rules
bind as much as the steps.

- **Assume the repo is healthy. Do not run the test suite, any single test,
  diagnostics, or timing scripts of your own.** They take minutes and, worse,
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
  `tools/`; the project's design documents; and this guide.

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
