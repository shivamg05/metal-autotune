# Measurement and log reference

Start with [the usage guide](usage.md) to configure a run. This page explains
internal measurements, installation methods, and diagnostic events.

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
is three times the accounted work. Every pause, whether slept or scheduled,
also returns the memory MLX keeps for reuse (`cache_cleared_gb`); left alone it
grew to 14.5 GB on a 24 GB Mac, and under that memory pressure timed passes ran
up to 7x slow. Each paired block (`sample_group`) logs every raw sample plus MLX
memory and swap in use before and after, so a block that ran under memory
pressure is visible. `cooling_scheduled` means a completed
comparison or correctness check has returned its verdict and CPU work may use
that cooldown. Candidate workers also return their remaining cooling deadline
to the parent (`cooling_adopted`), so process cleanup, preparation and judge
thinking can use the same interval. The parent waits before the next worker or
model evaluation; `cooling_reused` reports that remainder.
A pending cooldown does not mean the verdict is still being measured.
Step timing reuses a successful post-cooling warm-up instead of warming twice.
Its median is printed and logged as `step_clock` in `session.jsonl` before
cooling; the workload-level summary follows when the clock returns.
`ladder_result.result.detail.pacing` records accounted work and time spent
sleeping inside each successful worker phase. Parent waits appear in
`session.jsonl`. These are wall-clock accounting figures, not direct GPU-active
time: compilation and mixed CPU/GPU checks are still charged conservatively.
Warmup entries include their first-call time and total work; `off_clock_work`
records capture/correctness phase time. 
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
or one that reaches state it was not handed. Replay serves only the cache
positions the job recorded. Graph delivery also serves other positions and
cache sizes of the same kind of cache: it re-traces the scope once per new
cache state (at most 8 per scope) and there substitutes only kernels marked
`position_free` when bound, those whose inputs are computed from the scope's
arguments and own weights alone, and only where their verified operations
appear with their verified input shapes; if one does not, the call runs the
original. Timing still covers only what the job measured.
A `graph_fallback` row names why a scope moved to replay, `delivery_settled`
lists the split before the search, and `graph_verified`
records that both the inspected and the compiled calculation contained exactly
the expected substitutions. Every method passes the same correctness and
whole-model timing checks, and the artifact carries whichever was measured.

With `check_optimizations_for` in the manifest (automatic for an MLX-LM prompt
read from an empty cache: every prompt length from 3 to one past MLX-LM's
prefill chunk, logged as `other_sizes_default`), a stage after the search
replaces each changed
scope's delivery with generated code that reads its size from an argument
(`S_<scope>` classes), when the scope's replay at the primary size and at the
recorded range ends and middle differs only in integers that are one exact
linear function of that size. `serve_refused` names each scope that does not
generalize; it keeps its searched delivery. The generated code runs the scope's
recorded operations and each kernel inside an `if lo <= _n <= hi` of its own,
as plain MLX calls: nothing is traced, rewritten or compiled at run time, so a
new size costs nothing extra. It is not compiled because `mx.compile` builds a
new graph for every input shape; on seven models its gain was nil for the
transformers and large only for recurrent ones, while each new size paid for a
compile. Every other call goes to the original module. Each size is served only after
these checks:

- **Every size, no GPU work:** the model runs once per size in the range with
  each scope checked in place. The generated code (kernels off) and the
  original module build MLX graphs from the same arguments, and the graphs,
  outputs and state writes alike, must be the same operations, attributes,
  constants and sharing. A scope serves the contiguous sizes around the
  primary size that pass (`scopes` in the report's `check_optimizations_for` section, with the
  first difference found).
- **Every size, in the sandbox:** each bit-exact kernel (the `preserving` rule)
  is checked against its recorded operations with shader validation on,
  outputs poisoned and three runs bitwise equal. Inputs are the recorded
  boundary inputs cut or repeated to the size. This sweep takes no cooling
  pauses; nothing in it is timed.
- **A grid of about ten sizes, then twelve random sizes, in the sandbox:** the
  region clock's paired loop and relative margin (1% or three standard
  deviations) decide where the kernel wins or clearly loses. It is served out
  to the furthest winning size on each side of the primary, stopping at a grid
  point it does not win at and never reaching a size it lost at; the random
  sizes, drawn inside the range the grid picked, catch losses between grid
  points. `serve_kernel` logs each kernel's range, grid, samples and wrong
  sizes.

Kernels accepted within tolerance, starters, and kernels whose operations take
size-dependent settings keep their recorded size. The final check then runs
on the served model, also at the recorded serve sizes; if it fails or does not
confirm a win, `serve_withdrawn` records why, the searched wrappers go back,
and the final check runs again on them.

The default requested baseline is the compiled model. Under library
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

Candidate workers have a separate ten-second limit for each GPU evaluation,
including the first compilation and launch. Cooling does not consume that
limit; it still counts toward the overall worker budget. If either deadline
expires, the worker is killed. Once it exits, a fresh worker must complete a
small checked GPU operation, then wait until a fixed matmul runs within 1.5x of
its time recorded before the first candidate: killing a worker does not stop
GPU work it already submitted. Success rejects only the candidate and schedules
cooling before search continues. A wrong answer, or a GPU still busy after two
minutes, stops the job.

Keep other GPU work quiet. Interleaved measurements reduce drift, but cannot
guarantee that arbitrary background load affects both arms equally. The macOS
GPU utilization counter does not measure remaining throughput; compare it
with measured bandwidth, compute throughput, and timing noise.
Bandwidth is measured as read speed (1 GB read four times per sample, best of
six), because a model step's memory traffic is almost all reads. On an M4 it
reads about 106 GB/s when quiet. Other GPU work lowers that reading and, with
it, every memory-bound room estimate: a decode run started while the GPU was
79% busy measured 75 GB/s and reported 0% room.
The CLI prevents a second CLI job from running at the same time. If the A/A
control finds a significant difference between identical code twice, the
run stops before search because those measurements could create false wins.
