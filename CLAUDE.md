# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An autonomous optimizer for MLX models on Apple Silicon. It records the exact op calls a model runs on declared workloads, cuts that record into regions (runs of consecutive ops that one Metal kernel could replace), and runs a search loop: an LLM judge proposes one small kernel edit at a time, and a measurement harness compiles it, checks the model's outputs are unchanged, times it, and keeps only edits that are both correct and faster. The job ends with an artifact (kernels, a generated patch module, a swap table, apply(), report) that speeds up a freshly loaded model with none of the search machinery present.

## Source of truth

1. `DESIGN_SPEC.html` is the authority on WHAT the system does and owns the project vocabulary. Where anything in this repo disagrees with it, the spec wins. That includes the implementation plan and this file.
2. `IMPLEMENTATION_PLAN.md` is the authority on HOW to build it: stack, module boundaries, data model, measurement laws, milestones M0 through M12, and concrete defaults for the spec's loose words.

The spec ranks above the plan. The plan is a starting hypothesis for how to implement the spec, and it is expected to change as building teaches us more. The spec is the verified statement of where this project is going, and it changes rarely and deliberately.

Read the spec end to end before working on the loop, the ladder, regions, bind, or measurement. Use its vocabulary everywhere: code, comments, logs, and conversation.

## Vocabulary

- **manifest**: the job contract. Model path defining `build()`, workloads, sweep policy, tolerances, budget. Dtypes and quantization are not in it because they are not knobs.
- **workload**: the real input shapes to optimize for. A dim named in a shape (an "L") is sweepable; integer dims are model constants and never move.
- **trace**: the recorded op-call stream, per workload. Pass 1 records lazily with the patch surface installed; pass 2 is the step clock with the recorder fully removed.
- **region**: a run of consecutive recorded calls one kernel could replace, plus its boundary inputs/outputs and live values. All copies of the same op sequence across the model are one region.
- **roofline**: a region's physical speed limit from boundary bytes (never intermediates), flops, and launch count. Whichever term binds is recorded as `bound`: memory, compute, or launch.
- **scaffold**: the correct starting kernel the harness builds (stitched from MLX's shipped MSL where possible, naive lowering otherwise). Must pass the ladder before the judge may edit it; slow is normal.
- **hypothesis**: one small proposed kernel edit. The judge plans a queue in English and writes Metal only for the front ready item, one at a time.
- **the ladder**: the nine gates in order (static checks, compile, poison, watchdog, smoke, all workloads, shape sweep, determinism + numerics, region ship clock). First failure stops.
- **head / shipped**: the region's two bookmarks. Head is the best correct kernel being edited, even if slower. Shipped is the best correct-and-faster one; only shipped versions install.
- **bind**: generating a wrapper module that replays the region's scope with the cut replaced by the kernel, and swapping it in for the module on the live model. Gated by identity certification (the same replay with no kernel change must be invisible), verified by a literal retrace, then e2e; a miss swaps the original back.
- **the artifact**: `kernels/` + `patch/wrappers.py` + `swap_table` + `apply()` + `report`, loadable in a fresh process with no harness.

## Hard laws

- Dtypes and quantization stay frozen. Faster-but-wrong is discarded. Correct-but-slower never ships.
- The judge sees metadata only: never tensors, weights, activations, or tolerance values. It never overrides a failed check and never grades its own work. The harness owns every kernel call site, so `init_value`, `math_mode`, and streams are not the judge's to set.
- The baseline is chosen by measurement, per job: time the step both plain, exactly as `build()` hands the model, and under harness-applied `mx.compile`, and the faster one becomes the baseline. Both clocks and the choice go in the report. Tracing always records the plain model.
- No CPU fallback for GPU-dependent logic. If the environment cannot measure, raise.
- Every kernel evaluation runs out of process (Metal reads `MTL_SHADER_VALIDATION` at process launch; killing the process is how a wedged GPU recovers).
- The measurement laws in plan section 6 are invariants, not conventions: pair and interleave every comparison, duty-cycle pacing, warm until stable, medians for comparisons and running max for peaks, no absolute-time vetoes, defeat laziness in every timed loop.

## Architecture

Three components, kept strictly apart:

- `autotuner/`: the harness. Tracing (`trace/`), region cutting and pricing (`regions/`), measurement (`measure/`), scaffolds (`scaffold/`), the ladder (`ladder/`), subprocess sandbox (`sandbox/`), judge client (`judge/`), bind verification (`bind/`), the loop (`loop.py`), artifact emit (`artifact/`).
- `autotuner_runtime/`: a separate package that imports nothing from `autotuner`. It holds `apply()`, kernel loading, and the module-swap installer; the wrappers themselves are generated per job, and the harness installs the same generated code it measured, so the thing measured is the thing shipped. It is vendored into every artifact.
- The judge is stateless per call behind a strict JSON boundary; the queue and verdict log are its only memory. `judge/scripted.py` is a deterministic fake, and all loop/ladder tests run against it. No live-LLM tests.

Dependency direction: `judge/` may import `regions/` types; nothing imports `judge/` except `loop.py`. `sandbox/worker.py` is the child's entry point; `ladder/child.py` holds the gate bodies and imports only what one evaluation needs.

## Stack and commands

Python 3.12, `uv`-managed venv. Dependencies: `mlx` (exact version pinned in `pyproject.toml` and recorded in every report), `pyyaml`, `pytest`, `anthropic`.

```bash
uv sync
uv run pytest                          # full suite
uv run pytest tests/test_foo.py -k bar # one test
uv run autotune run manifest.yaml                     # the CLI (API-key judge)
uv run autotune run manifest.yaml --judge claude-cli  # judge on the local Claude Code login
uv run autotune run manifest.yaml --judge agent       # a live agent answers judge_io/ (AGENT_JUDGE.md)
```

`RUNNING.md` is the operator guide for running a job; it is user-facing and must
never reference the internal ledgers (PLATFORM.md, UNVERIFIED.md).

## Repo discipline

- Build in milestone order (plan section 9). M0 spikes the existential platform facts (patchability, bind mechanics, `metal_kernel` behavior, measurement floors) before anything depends on them; spike scripts live in `spikes/`.
- `PLATFORM.md`: every verified platform fact, with the spike that proves it. Each fact also gets a pinned test so an mlx upgrade fails loudly.
- `UNVERIFIED.md`: every code path not yet executed on real hardware, and why. Delete entries as they run.
- Spec-fixed constants (plan 5.12) may never be made configurable. Plan defaults are tunable, and every value actually used is recorded in the report.
- Tests are the fixture zoo (tiny models, each exercising one mechanism) and the cheat zoo (bad kernels, each asserting WHICH gate catches it). The harness's job is rejecting bad kernels, so its tests are bad kernels.

## Working style (from the maintainer)

- Verifiability is a core tenet of all work in this repo. Every output, whether a response, code, or a comment, must be reviewed so that no part of it is left unverified, either through experiments or external research. If something is found unverified, verify it, and do not stop until it is confirmed correct or incorrect. Let the result change the response or the code.
- Speak simply and precisely, in responses, code, and comments, like explaining to a colleague with no context on this work. Every sentence must carry value that is easy to take in. Verify this before writing anything.
- Code must be behaviorally correct, minimize the diff, and be intuitively organized: lean, clean, concise, clear. Write each diff so it can be verified correct with real environment or integration tests. Never stop at an incorrect implementation of any piece.
- Comments are sparse, one line where needed, never with em dashes. Each comment line must earn its place and sit where it belongs.
- Unit tests are thorough across edge cases and scenarios. Think from a user's perspective: what would they use this tool for, and how is that supported?
