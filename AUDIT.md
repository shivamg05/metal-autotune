# Audit, 2026-09-01

This is the problem list for the refactor that starts at the snapshot commit
4fb6e64. Everything here was checked against the code by reading it, and the
run numbers come from the five real runs in work-2026-08-31-*/RUN_FINDINGS.md.
The plan at the end is updated as commits land on the refactor branch.

## Size

The repo is about 22,000 lines, not 180,000. The harness is 9,048 lines, the
small package that ships inside artifacts 420, the tests 6,954, the platform
experiments 3,490, and the docs 2,440. The full test suite takes twelve
minutes, of which ten are three tests that each re-run a whole job.

## What actually blocks the purpose

The purpose is one thing: install a proven speedup on a real model. Five real
runs have not done it once, and every loss was on the harness's side. Ranked
by how directly each stops that first win:

1. **The judge never hears a failure.** The loop pops the next hypothesis
   before it tells the judge what happened to the last one. When the first
   item fails and the rest of the plan waits on it succeeding, nothing is
   ready, and the region closes with the judge never told. Five of sixteen
   regions in the last run ended this way with four to nine planned ideas
   untouched. The spec's own recovery path, "insert a fix that names the
   failure", cannot happen.
2. **The judge cannot make a small edit to a multi-op kernel.** The starting
   kernel for a multi-op region stages values through scratch buffers, but the
   judge has no way to declare them, so any edit that keeps them fails the
   first check on output count. It must delete all staging in one step, which
   is the opposite of "one small edit". Cost last run: four hypotheses and
   both scaffold fixes.
3. **The judge is briefed with a fraction of the contract.** The spec says it
   sees shapes, cost, the limiting resource, the source it is editing, the
   last verdict, distances from the physical limit, and the menu. The live
   payload omits the launch expression language, the menu descriptions, the
   rules it is graded on, the library time it must beat, the physical ceiling,
   and the source of a kernel that just failed. The renderer that carries all
   of that exists and has no caller. Hyphenated ids turning into invalid C
   names, and a 4,729-line library header the judge must echo back verbatim,
   are the same problem in practice.
4. **The harness fails its own correct kernels.** A stress test multiplies
   every float input by a thousand, quantization scales included, so the fp16
   reference overflows to infinity and correct kernels fail an exact-pattern
   rule. Five of sixteen regions were skipped at the starting kernel.
5. **Fourteen of sixteen regions were priced against a library that does not
   exist.** The model's SwiGLU is compiled into one fused kernel, but the
   tracer records it as three plain ops and every clock replays them as three
   kernels. Four regions lie entirely inside it and their headroom is pure
   phantom. The spec says a compiled section records as one opaque call. This
   is a correctness problem: a kernel that beats the phantom would add a
   launch the real model never pays.
6. **Two of the ranked regions could never be one kernel.** A chain where a
   matmul consumes an earlier matmul's output needs a grid-wide barrier, which
   one Metal launch does not have, and the determinism rule forbids the atomic
   tricks that fake one. Those two chains took ranks one and four and a fifth
   of the job's wall clock.
7. **After a first win, every later win would be rolled back.** The check
   that the cut became one custom dispatch compares against the job-start
   recording with only the new region's cut, so a second region's retrace has
   one custom node too many. Also, two copies of a region inside the same
   module overwrite each other during one install. The spec's economics
   ("0.3% times 32 copies") depend on both working.
8. **Close rules and the report use the wrong units.** One rule compares a
   per-copy gap to one percent of the whole step (closed six regions early
   last run, one with about a fifth of the step still on the table). The
   report's per-region speedup divides an all-copies total by a one-copy
   time. The terminal headline subtracts the job-start clock from the end
   clock, the exact drift number the paired comparison was added to replace;
   two runs printed phantom results with nothing installed.
9. **A second win is never compared to the first.** A kernel faster than the
   library but slower than the one already installed replaces it.

On the question of whether the judge should choose regions instead of the
rule-based enumeration: no. None of the five runs failed for want of a good
candidate. The list already held the fusions an engineer would write by hand
(gate and up projections with the SwiGLU epilogue, fused QKV plus rope, rope
plus attention). What killed them was the harness. And once the clock was
honest, every reachable region on single-token decode sat at the
weight-streaming wall, which no chooser, human or model, can move. The
enumeration needs two physics rules, not a new author.

## The full inventory

### The judge's side of the loop

- The cycle delivers each verdict one call late (item 1 above).
- No way to declare scratch outputs (item 2).
- Item ids become C symbols without sanitizing, and the judge is not told.
- A failed kernel's source is never sent back, so "fix it" is unwritable for
  a stateless judge; the parent id it names is never checked, which silently
  defeats the per-family strike count.
- The launch grammar, menu, laws, library time, ceiling, and family state
  never reach the live judge; one workload's shapes are sent when the region
  fires in several.
- The starting kernel is never timed, so the judge plans its first move
  without knowing whether it starts at 1.2x or 20x off the library.
- A proposal that omits the header or template gets an empty one, dropping
  the parent's; the stitched kernels' 140 KB header is sent on every call and
  cannot fit in the API transport's reply budget.
- Nothing the judge is sent or answers is written to disk for the API and
  Claude Code transports, and the Metal of every non-shipped kernel is lost
  at exit. The last two run write-ups had to reconstruct the judge's plan
  from gate strings.
- After a transport error the next request carries a stale verdict, and a
  yield is reported as an empty queue.

### The harness's own checks

- The scaled-up stress regime overflows fp16 references and perturbs weights,
  which are constants of the frozen model (item 4).
- A wrong output-shape expression passes the static check and crashes the
  child, returning a "subprocess" verdict with a traceback instead of a named
  gate, and it does not count toward the fail-streak rule.
- The shape sweep gate never receives anything to sweep inside a job: the
  manifest's sweep sizes are parsed and unused, and the log still records
  "sweep" as passed.
- The whole-model check has only the order-preserving path; a legitimately
  reordered kernel (split-K, retiling) would be judged against the wrong floor
  at the whole-model step.
- The child timeout is a fixed 120 seconds while the timing child's work
  grows with the region, so large regions would time out as "subprocess"
  failures.
- The fp16 default tolerance admits about ten ulps plus 0.02 absolute, far
  from the spec's "almost bit for bit" for order-preserving edits.
- Nothing checks that a Metal GPU exists or is idle before a job starts; the
  degraded-machine check runs after the most expensive clock and only warns.

### Regions and pricing

- Compiled sections recorded as plain ops (item 5). The library's own
  pre-compiled activations are substituted the same way.
- Chains grow through dependent matmuls (item 6).
- Copy grouping ignores weight shapes, so the 225-copy quantized matmul
  region mixes five different projection shapes and is priced, checked, and
  described at one of them.
- The physical-limit launch term counts the library's launches instead of the
  fused kernel's one, so a launch-bound chain can never show room.
- The coverage diagnostic sums nested cuts of the same block (586 ms against a
  52 ms step).
- Regions are captured and priced before anyone checks a starting kernel can
  be built; the first prefill run captured 4.3 GB and priced 22 regions, then
  skipped all 22 for lack of a scaffold.
- The chain anchors come from the module tree, which the spec says plays no
  part in cutting; they are load-bearing on this model and add view-led
  duplicate candidates.

### Measurement on a volatile machine

The paired clocks are right: comparisons alternate, report a ratio and an
agreement score, pricing measures each region as a share of the step in one
window, the ship clock re-measures the library beside the candidate, and the
end-of-job headline is paired. What remains:

- The two close rules that compare against the physical limit mix a per-copy
  time from the timing child with the job-start limit and the whole step
  (item 8).
- Pricing pays 51 full steps per region with pacing idles; at prefill scale
  that is nine minutes per region and three hours per job. The pair count for
  pricing does not need the ship clock's precision.
- Pacing degenerates when one sample exceeds the quarter-second chunk: every
  sample then pays two unmeasured full steps on top of the idle. At prefill
  the step clock spent 314 seconds to time 24 seconds of samples.
- Peaks are one two-second burst at job start, never recalibrated, with
  plausibility floors loose enough to miss a 2x under-read seen between two
  same-day runs.
- The A/A null control runs but its verdict is never checked; weight sharing
  between the two whole-model arms is counted but never verified.
- The plain-versus-compiled baseline race required by the plan has no
  implementation; every report carries an empty baseline.

### Installing a win

- Later ships fail verification and same-scope copies overwrite each other
  (item 7).
- A re-ship on the same span leaves the superseded kernel in the wrapper's
  kernel list, the swap table, and the artifact.
- No whole-model check runs after the last region, and the job never loads
  the artifact it wrote.
- The generated wrapper's fallback path bakes traced shapes as literals, so
  with a named dimension the fallback is wrong at other sizes.
- After a family is abandoned the head bookmark resets but its time does
  not, so a close rule can fire on a kernel that is no longer head.

### Logging and reporting

- Verdict rows omit the hypothesis text and the ship clock's library time,
  win, and sigma, which reach the parent process and stop there.
- No per-region summary at close: budget used, head time, outcome tally.
  Babble, transport errors, the scaffold, and scaffold fixes never enter the
  report's hypothesis list.
- Nothing renders the run for a person; the operator guide asks them to
  translate raw rows by hand.
- The log mixes per-copy and all-copies milliseconds under unlabeled names;
  the report is 78 percent one repeated stranded reason; the run log is not
  valid JSON when a detail holds infinity.

### The test suite

- Three whole-job tests at 4096x1024 exist to check rollback branches that
  need no measurement; they are 620 of the suite's 732 seconds.
- The health gates skip silently: the green run on 2026-09-01 skipped every
  timing-sensitive test including the only end-to-end proof that a win
  ships, and the summary line could not show it.
- The headline planted-win test asserts on the before-minus-after
  subtraction the harness itself now disowns.
- Thirteen sandbox children exercise an evaluation path the product never
  calls, including a ten-second hang by construction.
- Six files copy the same module-global tracer and end with an order-dependent
  "uninstall last" test.

### Leanness and organization

About 460 of the 9,468 harness lines can go with no behavior change: a
second sandbox evaluation path, a second prompt renderer, a test-only weight
binding path, a test-only region clock, canned judge scripts shipped in the
package, and a dozen small dead symbols. Five copies of the array-spec index,
four array-tree walkers, and two copies of the loop-sizing estimator fold
into one each. Every gate body lives in one 296-line function in the sandbox
worker while the ladder package holds a dead name tuple. The loop file
carries the job driver, the region loop, and the install orchestration in 870
lines.

### Writing

No em dashes anywhere. The dominant defect is citation by number: 88 comment
or docstring lines in 44 of 55 modules point at plan sections, law numbers,
spike slugs, milestone tags, audit findings, or dated decisions, and 40 of 55
module docstrings open that way. Twenty-seven comment blocks run three or
more lines; sixty docstrings run six or more. Three passages are wrong rather
than hard to follow: the watchdog constant is called spec-fixed while set to
twice the spec's number, the worker still says "the 10x rule", and the agent
guide says an unanswered judge request kills the job when the region closes
and the job continues. UNVERIFIED.md still says no live-judge call has
happened inside a job; the last run made 25.

### Where the implementation departs from the spec

| spec section | state |
|---|---|
| manifest | matches; two keys the spec does not name (primary, low/high) |
| trace, two passes | matches, except compiled sections record as plain ops |
| regions and the four rejection rules | matches in shape; "mixes CPU and GPU" is unimplemented; needs the dependent-matmul rule |
| pricing and ranking | paired share now; coverage line wrong; floor and headroom filters fine |
| roofline | launch term counts the wrong launches |
| opening a region, scaffold | matches; scaffold never timed |
| what the judge sees | a fraction of the contract |
| queue and conditionals | queue right; cycle delivers verdicts late |
| the nine gates | gate 7 never sweeps a named dim; stress regime overflows |
| verdicts | matches |
| bind and e2e | one ship per workload; no reordered-math whole-model path |
| closing rules | two rules wrong units; family-abandoned close missing; diminishing-ships rule can never fire |
| leaving a region | no re-recording or re-pricing after a close |
| artifact and report | speedup field wrong units; every kernel in the artifact even when superseded |

## Decisions

Made in this refactor, each open to veto:

- Compiled sections record as one opaque call, as the spec says, and the
  replay of a scope that contains one calls the compiled function by its
  import path so the scope stays installable.
- A chain ends before a matmul that consumes an earlier matmul's output in
  the same chain. This is a fifth rejection rule; it follows from the
  determinism gate, which forbids the atomics a one-launch split would need.
- Copies of a region with different weight shapes are different regions.
- The starting kernel is timed once, so the judge knows where it starts.
- Whole-job tests shrink to one real job; rollback paths are tested directly.
- A generated wrapper hands any call whose entering shapes differ from the
  recorded ones to the original module. Its replay bakes reshape targets and
  slice bounds as literals, so it is exact only at the recorded shapes; this
  makes an artifact correct at every size and faster at the primary one.
- A candidate copy that touches a shipped cut is covered, whether it lies
  inside the cut, contains it, or straddles its edge. Until the loop re-prices
  survivors on the patched model, a cut reaching past a shipped kernel has no
  honest library clock to be judged against.
- The hypothesis kind is the judge's own short label (you asked for this on
  2026-09-01): the seven menu kinds are suggestions, the laws are the only
  limit, and nothing in the harness ever keyed on the kind. Every call now
  carries worked examples (autotuner/judge/examples.py) whose replies pass
  the validator in a test, plus a short list of where wins come from.
- Every timed loop chains its passes and rotates a cache-defeating working
  set (the first Qwen run that reached the judge showed the region clock
  reading a one-threadgroup kernel ten times faster than it ran in the model,
  because Metal overlaps independent launches and three saved sets stay in
  cache). The chain link's cost is measured and subtracted. The roofline
  close rule is checked before the seed call, so a region already at the
  limit costs no plan.
- The baseline is the compiled model by default (you asked for this on
  2026-09-01, after mlx-metal-kernels found compile faster), with
  `baseline: plain` in the manifest as the alternative. It applies everywhere
  a win is measured: the step clocks, pricing's step and replay arms, the
  child's library arm, the veto, and the headline. Both step clocks are
  recorded. Choosing by measurement, as the spec says, is still to build.
  A step that keeps arrays in Python state (a KV cache) cannot be compiled
  from outside the model, since mx.compile swaps only state handed to it in a
  dict or list; the job detects that from the trace and takes the plain
  baseline with the reason recorded (found by the first Qwen run).

Left for you:

- The watchdog: the spec says ten times the library time, you ruled twenty
  on 2026-08-31, the code enforces twenty. I did not touch the spec.

## The plan

Each step is one commit on the refactor branch, verified by the tests it
names, in this order.

- [x] 1. Logging: hypothesis text and the ship clock's numbers in every
      verdict row; a candidates log with one line per attempt; the judge's
      requests and replies on disk; every evaluated kernel on disk; a
      per-region summary at close; valid JSON.
- [x] 2. Proposals: scratch outputs; ids sanitized and validated; parent
      resolved, header and template inherited, the library header kept on
      the harness side; output shapes checked statically.
- [x] 3. Briefing: one renderer with the whole contract, shapes per workload,
      the failed kernel's source, a legend for every key; the starting kernel
      timed.
- [x] 4. The cycle: verdict first, then pop; close rules in one frame; a
      second win must beat the installed one; report and headline in the
      same window.
- [x] 5. Checks: stress regime never touches weights and picks the largest
      finite scale; output-shape mistakes fail a named gate; child timeout
      follows the region's size.
- [x] 6. Regions: opaque compiled calls with import-path replay; the
      dependent-matmul rule; weight shapes in the fingerprint; scaffold
      coverage screened before pricing; launch term and coverage fixed.
- [x] 7. Installing: verification against every installed cut; same-scope
      copies in one wrapper; superseded kernels dropped; whole-model check
      and an artifact load at job end.
- [x] 8. Measurement: pricing at eight pairs with the step warmed once;
      ramp-warm bounded by work; peaks and the null before the step clock,
      raising on a machine that cannot measure; the null judged; weight
      sharing verified.
- [x] 9. Tests: one real job, rollback paths tested directly, skip reasons
      and durations printed, the second sandbox path retired, one shared
      tracer fixture.
- [x] 10. Leanness: dead code out, duplicate helpers folded, gate bodies in
      the ladder package, the loop split by stage.
- [x] 11. Writing: comments and docstrings without citations; the operator
      guide, the agent guide, the ledgers, and the plan brought back in line.
- [ ] 12. Remaining spec gaps, in this order as time allows: the shape sweep
      inside a job (done: traced, captured, checked by gate 7, and the wrapper
      hands unrecorded shapes to the original module), the plain-versus-compiled
      baseline (done as a manifest choice, compiled by default; the measured
      choice remains), the reordered-math whole-model check, re-pricing after a
      close.
