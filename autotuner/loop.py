"""The job driver and region loop (spec "The loop" and "Closing a region").

One hypothesis at a time. The harness owns measurement, correctness, bind, and
the artifact; the judge proposes. Verdicts: failed, correct-but-slower (climb),
tentative ship, then bind and e2e decide whether a ship is real.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from . import manifest as manifest_mod
from .bind.certify import certify_identity, find_scope_call, screen_scope
from .bind.emit import EmittedWrapper, NotReplayable, Splice, emit_wrapper
from .bind.swap import install as swap_install, uninstall as swap_uninstall
from .bind.verify import verify_retrace
from .e2e import _flatten_params, run_e2e, share_weights
from .judge.prompts import render_region_state
from .judge.queue import Queue, QueueError
from .judge.schema import JudgeBabble
from .ladder.gates import EvalSet, LadderJob, LadderResult, run_ladder
from .ladder.static_checks import RegionContract
from .artifact.emit import check_apply, emit_artifact, write_kernel
from .log import RunLog, TextLog, wall_now
from .measure.clocks import compare, step_clock
from .measure.controls import aa_null
from .measure.peaks import (BUSY_GPU_PERCENT, best_of, gpu_core_count, gpu_utilization,
                            implausible as peaks_implausible, measure_peaks)
from .measure.session import Session
from .regions.build import build_stretches, is_view, weight_like_ids
from .regions.fingerprint import group_copies
from .regions.price import PRICE_PAIRS, capture_boundaries, price_region
from .regions.rank import apply_floor, free_members, rank
from .regions.roofline import step_floor, stretch_roofline
from .regions.store import BoundaryStore
from .regions.types import Region, Stretch
from .report import Report
from .scaffold import uncovered_op
from .trace import Tracer
from .trace.recorder import ArrayRef
from .trace.walk import flatten_arrays
from .regions.sweep import SweepDivergence, locate_span
from .trace.serialize import nodes_to_json
from .trace.types import Trace
from .workload import materialize, workload_seeds
from autotuner_runtime.kernels import KernelSpec

MIN_WIN_MS = 0.030  # a win must save a few tens of microseconds per step across copies
CLOCK_PAIRS = 32    # ABBA pairs behind the ship clock and the headline


@dataclass
class RegionRun:
    region: Region
    scaffold: KernelSpec | None = None
    head: KernelSpec | None = None
    shipped: KernelSpec | None = None
    head_ms: float | None = None
    shipped_ms: float | None = None
    library_ms: float | None = None   # the library beside head, from head's own clock
    head_ratio: float | None = None   # head_ms over its library_ms: the drift-free number
    shipped_ratio: float | None = None
    head_floor_ms: float | None = None     # the floor probe clocked beside head, same child
    shipped_floor_ms: float | None = None
    head_tag: str = "preserving"      # assoc tag of the edit that produced head
    shipped_tag: str = "preserving"
    last_kernel: str | None = None    # the kernel the latest verdict was about
    attempts: dict[str, dict] = field(default_factory=dict)  # kernel id -> its verdict
    hypotheses: int = 0
    close_rule: str | None = None
    kernels: dict[str, KernelSpec] = field(default_factory=dict)


class JobRunner:
    def __init__(self, manifest_path: str | Path, work_dir: str | Path,
                 judge_factory, session: Session | None = None,
                 clock_pairs: int = CLOCK_PAIRS, refuse_degraded: bool = True):
        self.manifest = manifest_mod.load(manifest_path)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if (self.work_dir / "run.jsonl").exists():
            # run.jsonl appends; a reused dir would interleave two jobs' rows
            raise RuntimeError(
                f"{self.work_dir} holds a previous run (run.jsonl exists); "
                "pass a fresh --work-dir or move the old one aside")
        self.judge_factory = judge_factory
        self.clock_pairs = clock_pairs
        self.refuse_degraded = refuse_degraded  # a machine that cannot measure stops the job
        self.baseline = self.manifest.baseline  # settled by _clock_steps once the traces are in
        self.session = session or Session(log_path=self.work_dir / "session.jsonl")
        self.log = RunLog(self.work_dir / "run.jsonl")
        self.candidates = TextLog(self.work_dir / "candidates.log")
        self.kernel_dir = self.work_dir / "kernels"
        self.report = Report(manifest_path=str(manifest_path))
        self.store = BoundaryStore(self.work_dir / "boundaries")
        self.tracer = Tracer()
        self.traces: dict[str, Trace] = {}
        self.tensors: dict[str, list[mx.array]] = {}
        # the sweep: one trace and one input set per named dim size, keyed
        # "workload@dim=size", and each region's span located in that trace
        self.sweep_traces: dict[str, Trace] = {}
        self.sweep_tensors: dict[str, list[mx.array]] = {}
        self.sweep_spans: dict[tuple[str, str], Stretch] = {}
        self.step_ms: dict[str, float] = {}
        self.peaks = None  # measured by measure_machine before any clock
        self.total_hypotheses = 0
        self.lessons: list[dict] = []  # what the judge wrote down for later regions
        self.installed: dict[str, tuple] = {}  # scope -> (original module, splices, kernels)
        self.cuts: dict[str, dict[tuple[int, int], str]] = {}  # workload -> {span: kernel id}
        self.certified_scopes: set[str] = set()
        self.emitted: dict[str, EmittedWrapper] = {}
        self.final_ok = True

    # -- stage 1: model and traces -------------------------------------------

    def load_model(self):
        manifest_mod.check_build(self.manifest)
        self.tracer.install(model_module_name="autotune_model")
        spec = importlib.util.spec_from_file_location("autotune_model", self.manifest.model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.build()
        self.baseline_model = module.build()
        shared = share_weights(self.model, self.baseline_model)
        mx.clear_cache()  # return the freed second weight copy to the OS
        total = len(_flatten_params(self.baseline_model.parameters())) \
            if hasattr(self.baseline_model, "parameters") else 0
        self.log.append("model", shared_weights=shared, parameters=total)
        if shared < total:
            self._env_warning(f"only {shared} of {total} parameters could be shared between "
                              "the two model copies; both stay resident, which adds noise "
                              "to the whole-model checks")

    def trace_workloads(self):
        for w in self.manifest.workloads:
            seeds = workload_seeds(self.manifest.seed, w.name, manifest_mod.BOUNDARY_INPUT_SETS)
            self.tensors[w.name] = materialize(w, self.manifest.primary, seeds[0])
            trace, _ = self.tracer.trace(self.model, self.tensors[w.name])
            self.traces[w.name] = trace
            if trace.in_pass_evaluation:
                self.log.append("memory_warning", workload=w.name,
                                detail="model evaluated mid-record; intermediates stayed resident")
            self.log.append("trace", workload=w.name, nodes=len(trace.nodes))
            for label, dims in self._sweep_points(w):
                self.sweep_tensors[label] = materialize(w, dims, seeds[0])
                self.sweep_traces[label], _ = self.tracer.trace(self.model, self.sweep_tensors[label])
                self.log.append("trace", workload=label, nodes=len(self.sweep_traces[label].nodes))

    def _sweep_points(self, w) -> list[tuple[str, dict[str, int]]]:
        """Each named dim of the workload at each sweep size but the primary,
        the other dims held at their primary sizes."""
        points = []
        for dim in sorted(w.named_dims()):
            for size in self.manifest.sweep[dim]:
                if size != self.manifest.primary[dim]:
                    points.append((f"{w.name}@{dim}={size}", {**self.manifest.primary, dim: size}))
        return points

    # -- stage 2: regions -----------------------------------------------------

    def build_regions(self) -> list[Region]:
        stretches = {name: build_stretches(t, name) for name, t in self.traces.items()}
        regions = group_copies(self.traces, stretches)
        viable = []
        for r in regions:
            reason = screen_scope(self.traces[r.members[0].workload], r.members[0].scope_stack)
            if reason is not None:
                r.rejected = f"no certified delivery scope: {reason}"
            elif (op := uncovered_op(r.ops)) is not None:
                # nothing to edit if the harness cannot write a starting kernel;
                # say so before any capture or pricing is spent on it
                r.rejected = f"no scaffold for {op}"
            if r.rejected:
                self.report.stranded.append({"fingerprint": r.fingerprint, "ops": list(r.ops),
                                             "reason": r.rejected})
            else:
                viable.append(r)
        self.log.append("regions", candidates=len(regions), screened=len(viable))
        return viable

    def capture_and_price(self, regions: list[Region]) -> list[Region]:
        """Save every viable region's boundary tensors, then measure: the
        machine, the step, and each region's share of it."""
        self._capture(regions)
        # the recorder must be fully removed before any timing
        self.tracer.uninstall()
        mx.clear_cache()  # capture's activation buffers; warm-up refills what clocks need
        self.measure_machine()
        self._clock_steps()
        return self._price_and_rank(regions)

    def _capture(self, regions: list[Region]) -> None:
        """k input sets per workload, the boundary arrays of every viable
        region saved from a recorded pass at each."""
        for w in self.manifest.workloads:
            wanted: set[int] = set()
            for r in regions:
                for m in r.members:
                    if m.workload == w.name:
                        wanted |= set(m.input_ids) | set(m.output_ids)
            if not wanted:
                continue
            seeds = workload_seeds(self.manifest.seed, w.name, manifest_mod.BOUNDARY_INPUT_SETS)
            for si, seed in enumerate(seeds):
                tensors = materialize(w, self.manifest.primary, seed)
                arrays = capture_boundaries(self.tracer, self.model, tensors,
                                            self.traces[w.name], wanted)
                for r in regions:
                    for m in r.members:
                        if m.workload != w.name:
                            continue
                        try:
                            ins = {a: arrays[a] for a in m.input_ids}
                            outs = {a: arrays[a] for a in m.output_ids}
                        except KeyError as e:
                            # a capture gap costs this region, never the job
                            r.rejected = f"capture missed boundary array {e}"
                            break
                        self.store.save(r.fingerprint, w.name, si, "inputs", ins)
                        self.store.save(r.fingerprint, w.name, si, "outputs", outs)
                        break  # sets are per representative member; copies share the clock
            self._capture_sweep(w, [r for r in regions if not r.rejected])

    def _capture_sweep(self, w, regions: list[Region]) -> None:
        """One correctness set per sweep size for every region the workload
        fires in: the region's span located in the trace at that size, its
        boundary arrays saved from one recorded pass. A region whose ops the
        model no longer runs in one run at some size strands here."""
        for label, _dims in self._sweep_points(w):
            retrace = self.sweep_traces[label]
            wanted: set[int] = set()
            for r in regions:
                rep = next((m for m in r.members if m.workload == w.name), None)
                if rep is None:
                    continue
                try:
                    span = locate_span(self.traces[w.name], rep, retrace, w.name)
                except SweepDivergence as e:
                    r.rejected = f"the op stream diverges at {label}: {e}"
                    continue
                self.sweep_spans[(r.fingerprint, label)] = span
                wanted |= set(span.input_ids) | set(span.output_ids)
            if not wanted:
                continue
            arrays = capture_boundaries(self.tracer, self.model, self.sweep_tensors[label],
                                        retrace, wanted)
            for r in regions:
                span = self.sweep_spans.get((r.fingerprint, label))
                if span is None:
                    continue
                try:
                    ins = {a: arrays[a] for a in span.input_ids}
                    outs = {a: arrays[a] for a in span.output_ids}
                except KeyError as e:
                    r.rejected = f"capture missed boundary array {e} at {label}"
                    continue
                self.store.save(r.fingerprint, label, 0, "inputs", ins)
                self.store.save(r.fingerprint, label, 0, "outputs", outs)

    def _step_fn(self, model, tensors: list[mx.array], baseline: str | None = None):
        """One step of the model as the baseline runs it. Compiled means a
        fresh mx.compile closure, built here and never reused across a swap:
        compile caches on the callable, so an old closure keeps the old graph."""
        if (baseline or self.baseline) == "compiled":
            compiled = mx.compile(lambda *t: model(*t))
            mx.eval(compiled(*tensors))  # compile now, not inside a timed sample
            return lambda: compiled(*tensors)
        return lambda: model(*tensors)

    def _timed_arms(self):
        """The untouched and the patched model on the first workload, each as
        the baseline runs it, for the step-time veto."""
        first = self.tensors[self.manifest.workloads[0].name]
        return self._step_fn(self.baseline_model, first), self._step_fn(self.model, first)

    def _clock_steps(self) -> None:
        """The step clocks on every workload, and the baseline every share and
        win is measured against. A step that keeps arrays in Python state (a
        KV cache) cannot be compiled from outside the model: mx.compile swaps
        only state handed to it in a dict or list, and one compiled call would
        leave the model holding tracers. Such a step gets the plain baseline,
        and its compiled clock is never taken."""
        kept = {w: len(t.python_retained()) for w, t in self.traces.items() if t.python_retained()}
        requested = self.manifest.baseline
        self.baseline = "plain" if kept else requested
        reason = None if not kept else (
            "the step keeps arrays in Python state (" +
            ", ".join(f"{w}: {n}" for w, n in kept.items()) +
            "); a compiled call would leave the model holding tracers")
        self.report.baseline = {"requested": requested, "choice": self.baseline,
                                "compiled_available": not kept, "reason": reason, "clocks_ms": {}}
        self.log.append("baseline", requested=requested, choice=self.baseline,
                        compiled_available=not kept, reason=reason)
        for w in self.manifest.workloads:
            tensors = self.tensors[w.name]
            clocks = {"plain": step_clock(self.session, self._step_fn(self.model, tensors, "plain")).median_ms,
                      "compiled": None if kept else step_clock(
                          self.session, self._step_fn(self.model, tensors, "compiled")).median_ms}
            self.report.baseline["clocks_ms"][w.name] = clocks
            self.step_ms[w.name] = clocks[self.baseline]
            self.report.step_ms[w.name] = {"before": clocks[self.baseline]}
            self.log.append("step_clock", workload=w.name, phase="before", baseline=self.baseline,
                            median_ms=clocks[self.baseline], plain_ms=clocks["plain"],
                            compiled_ms=clocks["compiled"])
            if self.peaks is not None:  # a job measures the chip before it clocks; a test may not
                # the scout line: how much of this step is physics no kernel
                # can touch, and how much is room
                floor = step_floor(self.traces[w.name], self.peaks, self.step_ms[w.name])
                self.report.coverage.setdefault("step_floor", {})[w.name] = floor
                self.log.append("step_floor", workload=w.name, **floor)

    def _price_and_rank(self, regions: list[Region]) -> list[Region]:
        """Each region's share of the step, its physical limit, the floor and
        headroom filters, and the ranking."""
        step_fns = {w.name: self._step_fn(self.model, self.tensors[w.name])
                    for w in self.manifest.workloads}
        warmed_steps: set[str] = set()
        for r in regions:
            sets_for = {}
            weights = {}
            seen_w = set()
            for m in r.members:
                weights.setdefault(m.workload, {})
                if m.workload in seen_w:
                    continue  # captured sets carry the representative's ids only
                seen_w.add(m.workload)
                sets = [self.store.load(r.fingerprint, m.workload, si, "inputs")
                        for si in range(self.store.set_count(r.fingerprint, m.workload))]
                if sets:
                    sets_for[(m.workload, m.start_seq)] = sets
            price_region(r, self.session, self.traces, sets_for, weights, step_fns,
                         baseline=self.baseline,
                         pairs=PRICE_PAIRS, warmed_steps=warmed_steps)
        # a second look at the peaks now that the chip has been working: only
        # a higher reading can be truer, and the rooflines below use the best
        self._record_peaks(best_of(self.peaks, measure_peaks(self.session)), "after_pricing")
        for r in regions:
            rep = r.members[0]
            # one copy's cost against one copy's floor, both from one paired
            # window, so nothing the machine did between pricing and the peak
            # readings can open headroom that is not there
            r.roofline = stretch_roofline(
                self.traces[rep.workload], rep, self.peaks,
                t_orig_ms=r.t_rep_ms.get(rep.workload) or 1e-9,
                floor_ms=r.t_floor_ms.get(rep.workload),
            )
        kept = rank(apply_floor(regions))
        # every compute op belongs to exactly one atomic region, so their
        # shares sum to the fraction of the step that any candidate can reach;
        # the rest runs inside stranded scopes or ops no kernel can replace
        self.report.coverage.update({
            "share_of_step_inside_candidates": {
                w.name: sum(r.p.get(w.name, 0.0) for r in regions if self._atomic(r))
                for w in self.manifest.workloads
            },
            "step_ms": dict(self.step_ms),
        })
        for r in regions:
            if r.rejected:  # share floor and no-headroom cuts both belong in the report
                self.report.stranded.append({"fingerprint": r.fingerprint, "ops": list(r.ops),
                                             "reason": r.rejected})
        self.log.append("ranked", kept=len(kept))
        return kept

    def _atomic(self, region: Region) -> bool:
        """One compute op, with any views in front of it taken of weights: a
        projection through its transpose, or a lone op. Each compute op is in
        exactly one such region."""
        rep = region.members[0]
        trace = self.traces[rep.workload]
        nodes = trace.nodes[rep.start_seq:rep.end_seq + 1]
        if sum(not is_view(n) for n in nodes) != 1:
            return False
        weight_like = weight_like_ids(trace)
        return all(all(a in weight_like for a in n.in_arrays) for n in nodes if is_view(n))

    def measure_machine(self) -> None:
        """The chip's own limits, and whether this machine can measure at
        all: a busy GPU or a reading no healthy chip gives stops the job when
        refuse_degraded is set, since every roofline and every absolute clock
        would describe a sick machine."""
        busy = gpu_utilization()
        self.report.session["gpu_utilization_at_start_pct"] = busy
        if busy is not None and busy > BUSY_GPU_PERCENT:
            self._cannot_measure(f"the GPU is {busy:.0f}% busy before this job has issued any "
                                 "work; another process is using it")
        self._record_peaks(measure_peaks(self.session), "job_start")
        reason = peaks_implausible(self.peaks)
        if reason:
            self._cannot_measure(reason)
        floor = aa_null(self.session, pairs=8)
        self.report.session["aa_floor_sigma_ms"] = floor.sigma_ms
        self.log.append("aa_floor", sigma_ms=floor.sigma_ms, median_delta_ms=floor.median_delta_ms,
                        stability=round(floor.stability, 3))
        if floor.wins_by(0.0) or floor.loses_by(0.0):
            self._env_warning(f"the A/A control found a {floor.median_delta_ms:+.4f} ms difference "
                              "between two runs of the same code; this session's noise is "
                              "not symmetric and small wins will be missed")

    def _record_peaks(self, peaks, when: str) -> None:
        self.peaks = peaks
        self.report.peaks = {"bandwidth_gbps": peaks.bandwidth_gbps,
                             "flops_gflops": peaks.flops_gflops, "launch_us": peaks.launch_us}
        self.log.append("peaks", when=when, **self.report.peaks)

    def _cannot_measure(self, reason: str) -> None:
        if self.refuse_degraded:
            self.log.append("job_refused", reason=reason)
            raise RuntimeError(f"this machine cannot measure right now: {reason}. Wait for it "
                               "to go quiet and cool, then start a fresh run")
        self._env_warning(reason)

    def _env_warning(self, detail: str) -> None:
        self.log.append("env_warning", detail=detail)
        print(f"WARNING: {detail}")

    # -- stage 3: one region --------------------------------------------------

    def _contract(self, region: Region) -> RegionContract:
        rep = region.members[0]
        specs = self.traces[rep.workload].span_specs(rep.start_seq, rep.end_seq)
        return RegionContract(
            input_names=tuple(f"in{i}" for i in range(len(rep.input_ids))),
            input_ranks=tuple(len(specs[a][0]) for a in rep.input_ids),
            input_dtypes=tuple(specs[a][1] for a in rep.input_ids),
            output_names=tuple(f"out{i}" for i in range(len(rep.output_ids))),
            output_ranks=tuple(len(specs[a][0]) for a in rep.output_ids),
            output_dtypes=tuple(specs[a][1] for a in rep.output_ids),
            live_outputs=tuple(f"out{i}" for i in range(len(rep.output_ids))),
            input_shapes=tuple(tuple(specs[a][0]) for a in rep.input_ids),
            output_shapes=tuple(tuple(specs[a][0]) for a in rep.output_ids),
        )

    def _eval_sets(self, region: Region) -> list[EvalSet]:
        sets = []
        seen = set()
        for m in region.members:
            if m.workload in seen:
                continue
            seen.add(m.workload)
            k = self.store.set_count(region.fingerprint, m.workload)
            if region.t_orig_ms.get(m.workload) is None:
                # open_region screens this; backstop so it can never reach
                # ladder validation as a bare None
                raise RuntimeError(f"region eval set unpriced for workload {m.workload!r}")
            sets.append(EvalSet(
                label=m.workload,
                inputs_paths=[str(self.store._path(region.fingerprint, m.workload, i, "inputs"))
                              for i in range(k)],
                reference_paths=[str(self.store._path(region.fingerprint, m.workload, i, "outputs"))
                                 for i in range(k)],
                t_library_ms=region.t_rep_ms.get(m.workload),
                correctness_only=False,
                nodes_json=None,
            ))
        for label, span, _specs in self._sweep_instances(region):
            nodes = self.sweep_traces[label].nodes[span.start_seq:span.end_seq + 1]
            sets.append(EvalSet(
                label=label,
                inputs_paths=[str(self.store._path(region.fingerprint, label, 0, "inputs"))],
                reference_paths=[str(self.store._path(region.fingerprint, label, 0, "outputs"))],
                correctness_only=True,
                nodes_json=nodes_to_json(nodes),
            ))
        return sets

    def _sweep_instances(self, region: Region) -> list[tuple[str, Stretch, dict]]:
        """(label, span, array specs) for each sweep size the region was
        located and captured at."""
        out = []
        for (fingerprint, label), span in sorted(self.sweep_spans.items()):
            if fingerprint == region.fingerprint:
                specs = self.sweep_traces[label].span_specs(span.start_seq, span.end_seq)
                out.append((label, span, specs))
        return out

    def _ladder_job(self, region: Region, kernel: KernelSpec, assoc_tag: str,
                    run_clock: bool) -> LadderJob:
        rep = region.members[0]
        trace = self.traces[rep.workload]
        nodes = trace.nodes[rep.start_seq:rep.end_seq + 1]
        contract = self._contract(region)
        float_out = next(
            (d for d in contract.output_dtypes if d in ("float16", "bfloat16", "float32")),
            "float32",
        )
        one_copy_ms = region.t_rep_ms.get(rep.workload) or 0.0
        return LadderJob(
            baseline=self.baseline,
            kernel=kernel,
            contract=contract,
            assoc_tag=assoc_tag,
            nodes_json=nodes_to_json(nodes),
            input_ids=rep.input_ids,
            output_ids=rep.output_ids,
            eval_sets=self._eval_sets(region),
            tolerances=self.manifest.tolerance_for(float_out),
            min_win_ms=MIN_WIN_MS / max(region.copies, 1),
            run_clock=run_clock,
            clock_pairs=self.clock_pairs,
            # the timing child runs a few hundred passes under pacing; a region
            # whose one pass takes hundreds of ms needs minutes, not a fixed cap
            timeout_s=120.0 + 0.8 * one_copy_ms,
            weight_inputs=tuple(a in trace.weights for a in rep.input_ids),
            compute_floor_ms=region.roofline.t_compute_ms if region.roofline else 0.0,
        )

    def _kernel_from_proposal(self, run: RegionRun, region: Region, proposal, hyp_id: str) -> KernelSpec:
        kid = f"r{region.fingerprint[:6]}_{hyp_id}"
        parent = resolve_parent(run, proposal.parent_kernel_id)
        return kernel_from_proposal(self._contract(region), parent, proposal, kid)

    def open_region(self, region: Region, judge) -> RegionRun:
        from .scaffold import NoScaffold, build_scaffold

        run = RegionRun(region=region)
        for m in region.members:
            if region.t_orig_ms.get(m.workload) is None:
                run.close_rule = f"workload {m.workload!r} was never priced for this region"
                self.log.append("region_skip", fingerprint=region.fingerprint,
                                reason=run.close_rule)
                return run
        rep = region.members[0]
        trace = self.traces[rep.workload]
        try:
            # the sweep sizes are extra instances, so a dim that moves across
            # them stays symbolic instead of baking in as the primary's literal
            scaffold = build_scaffold(trace, rep, [
                [specs[a][0] for a in span.input_ids]
                for _label, span, specs in self._sweep_instances(region)])
        except NoScaffold as e:
            run.close_rule = f"no scaffold: {e}"
            self.log.append("region_skip", fingerprint=region.fingerprint, reason=run.close_rule)
            return run
        except Exception as e:
            # a generator bug costs the region with a named reason, not the job
            run.close_rule = f"scaffold build error: {type(e).__name__}: {e}"
            self.log.append("region_skip", fingerprint=region.fingerprint, reason=run.close_rule)
            return run
        scaffold = self._rename(scaffold, region, "scaffold")
        run.kernels[scaffold.kernel_id] = scaffold
        write_kernel(self.kernel_dir, scaffold)
        result = run_ladder(self._ladder_job(region, scaffold, "preserving", run_clock=True))
        if result.outcome == "failed":
            self._record_attempt(run, "scaffold", "scaffold", "the harness's starting kernel",
                                 "preserving", scaffold, None, result)
            self.log.append("scaffold_failed", fingerprint=region.fingerprint,
                            gate=result.failed_gate, detail=result.detail)
            # one judge fix attempt, per the spec
            fixed = self._judge_fix(run, judge, scaffold, result)
            if fixed is None:
                run.close_rule = f"scaffold failed {result.failed_gate} (the one judge fix attempt did not produce a passing kernel)"
                return run
            scaffold, result = fixed
        run.scaffold = scaffold
        self._set_head(run, scaffold, result, "preserving")
        self.log.append("scaffold_ok", fingerprint=region.fingerprint,
                        kernel=scaffold.kernel_id, region_ms=result.region_ms,
                        library_ms=result.library_ms)
        outcome = result.outcome
        if outcome == "tentative_ship":
            # the harness's own kernel beat the library on the region clock;
            # it ships through the same install and whole-model checks as any edit
            outcome = "shipped" if self._bind_and_promote(run, scaffold, result) else "rolled_back"
            if outcome == "shipped":
                run.shipped, run.shipped_ms, run.shipped_ratio = scaffold, result.region_ms, _ratio(result)
                run.shipped_floor_ms = result.floor_ms
        repaired = scaffold is not run.kernels.get(f"r{region.fingerprint[:6]}_scaffold")
        self._record_attempt(
            run, "scafix" if repaired else "scaffold", "fix" if repaired else "scaffold",
            "the judge's one fix of the starting kernel" if repaired else "the harness's starting kernel",
            "preserving", scaffold, None, result, outcome=outcome)
        return run

    def _rename(self, spec: KernelSpec, region: Region, tag: str) -> KernelSpec:
        kid = f"r{region.fingerprint[:6]}_{tag}"
        d = {k: getattr(spec, k) for k in spec.__dataclass_fields__}
        d["kernel_id"] = kid
        d["name"] = f"at_{kid}"
        return KernelSpec(**d)

    def _judge_fix(self, run: RegionRun, judge, scaffold, result):
        """The spec's one repair attempt on a starting kernel that failed its
        own checks. Returns (kernel, ladder result) or None."""
        region = run.region
        run.head, run.last_kernel = scaffold, scaffold.kernel_id
        verdict = _verdict_payload("scaffold", scaffold.kernel_id, "failed", result)
        writing_for = {"id": "scafix", "kind": "fix", "assoc_tag": "preserving",
                       "hypothesis": "repair the starting kernel so it passes the checks"}
        try:
            resp = judge.next(self._meta(run, Queue(), writing_for), verdict)
        except Exception as e:
            self.log.append("scaffold_fix", fingerprint=region.fingerprint,
                            outcome="judge_error", reason=str(e)[:200])
            return None
        if resp.kernel is None:
            self.log.append("scaffold_fix", fingerprint=region.fingerprint, outcome="yield")
            return None
        fixed = self._kernel_from_proposal(run, region, resp.kernel, "scafix")
        run.kernels[fixed.kernel_id] = fixed
        write_kernel(self.kernel_dir, fixed)
        check = run_ladder(self._ladder_job(region, fixed, "preserving", run_clock=True))
        if check.outcome == "failed":
            self._record_attempt(run, "scafix", "fix", "the judge's one fix of the starting kernel",
                                 "preserving", fixed, resp.kernel.parent_kernel_id, check)
            self.log.append("scaffold_fix", fingerprint=region.fingerprint,
                            outcome="fix_failed_ladder", gate=check.failed_gate,
                            detail=check.detail)
            return None
        self.log.append("scaffold_fix", fingerprint=region.fingerprint, outcome="fixed")
        return fixed, check

    def hypothesis_cycle(self, run: RegionRun, judge) -> None:
        """One hypothesis at a time until the budget is spent. Every call to
        the judge carries the last verdict and the queue; the judge edits its
        plan first, then writes Metal for the front ready item; the ladder
        decides; repeat. A reply with nothing to evaluate (a yield, plan
        edits the queue refuses, a kernel for an item that is not ready) is
        asked again once with the reason; each further one costs an attempt,
        so the budget is spent by the judge and never handed back."""
        region = run.region
        queue = Queue()
        seed, failure = self._ask_judge(run, "seed", lambda: judge.seed(self._meta(run, queue, None)))
        if seed is None:
            run.close_rule = ("the judge babbled at seed" if failure == "babble"
                              else f"judge unavailable at seed ({failure})")
            return
        self._note_lesson(run, seed)
        try:
            queue.seed(seed.queue)
        except QueueError as e:
            run.close_rule = f"the judge's seed queue was inconsistent ({e})"
            return
        verdict = None
        errors = 0    # consecutive transport failures; three close the region
        refused = 0   # consecutive replies with nothing to evaluate

        while True:
            rule = self._close_rule(run)
            if rule:
                run.close_rule = rule
                return
            front = queue.peek_ready()
            resp, failure = self._ask_judge(run, "next", lambda: judge.next(
                self._meta(run, queue, _item_view(front) if front else None), verdict))
            hyp = front.id if front else "none"
            if resp is None:
                if failure == "babble":
                    # a babbling judge burns budget, so it exhausts its region
                    run.hypotheses += 1
                    self.total_hypotheses += 1
                    self._record_attempt(run, hyp, front.kind if front else "none",
                                         front.hypothesis if front else "", None, None, None,
                                         None, gate="judge_babble")
                    verdict = {"hypothesis_id": hyp, "outcome": "failed", "failed_gate": "judge_babble"}
                else:
                    errors += 1
                    self._record_attempt(run, hyp, front.kind if front else "none",
                                         front.hypothesis if front else "", None, None, None,
                                         None, gate="judge_error", reason=failure)
                    verdict = {"hypothesis_id": hyp, "outcome": "failed",
                               "failed_gate": "judge_error", "detail": {"reason": failure}}
                    if errors >= 3:
                        run.close_rule = f"judge unavailable (3 straight transport errors, last {failure})"
                        return
                continue
            errors = 0
            self._note_lesson(run, resp)
            item = None
            try:
                queue.apply_mutations(resp.mutations)
            except QueueError as e:
                problem = f"your plan edits were refused and none applied: {e}"
            else:
                if resp.kernel is None:
                    problem = (f"a yield is refused while the budget lasts: "
                               f"{self._attempts_left(run)} attempts remain, propose")
                else:
                    item = queue.pop_ready(resp.kernel.item_id)
                    problem = None if item is not None else (
                        f"your kernel names no ready item: queued {', '.join(queue.ids()) or 'nothing'}"
                        f", every one waiting on a verdict that has not come")
            if item is None:
                self.log.append("plan_refused", fingerprint=region.fingerprint, reason=problem)
                verdict = {**(verdict or {}), "plan_refused": problem}
                refused += 1
                if refused > 1:
                    # the one free re-ask is spent; every further empty reply costs an attempt
                    run.hypotheses += 1
                    self.total_hypotheses += 1
                    self._record_attempt(run, hyp, front.kind if front else "none",
                                         front.hypothesis if front else "", None, None, None,
                                         None, gate="plan_refused", reason=problem)
                continue
            refused = 0

            run.hypotheses += 1
            self.total_hypotheses += 1
            parent_spec = resolve_parent(run, resp.kernel.parent_kernel_id)
            parent = parent_spec.kernel_id if parent_spec else resp.kernel.parent_kernel_id
            kernel = self._kernel_from_proposal(run, region, resp.kernel, item.id)
            run.kernels[kernel.kernel_id] = kernel
            write_kernel(self.kernel_dir, kernel)
            if parent_spec is not None:
                result = run_ladder(self._ladder_job(region, kernel, item.assoc_tag, run_clock=True))
            else:
                # the parent is the judge's own memory of what it edited; an
                # unknown one is a mistake to name, not a kernel to measure
                result = LadderResult("failed", "static", {"failures": [{
                    "check": "parent_kernel_id",
                    "detail": f"{parent!r} names no kernel of this region; say head, "
                              f"scaffold, shipped, a hypothesis id, or one of {sorted(run.kernels)}"}]},
                    None, None, None, None, [])
            outcome = self._apply_verdict(run, item, kernel, result)
            queue.record_verdict(item.id, outcome)
            verdict = _verdict_payload(item.id, kernel.kernel_id, outcome, result)
            self._record_attempt(run, item.id, item.kind, item.hypothesis, item.assoc_tag,
                                 kernel, parent, result, outcome=outcome)

    def _note_lesson(self, run: RegionRun, resp) -> None:
        """A sentence the judge wrote for later regions of this job; every
        later call carries the list."""
        if resp.lesson:
            self.lessons.append({"region": run.region.fingerprint[:8], "ops": list(run.region.ops),
                                 "lesson": resp.lesson})
            self.log.append("lesson", fingerprint=run.region.fingerprint, lesson=resp.lesson)

    def _attempts_left(self, run: RegionRun) -> int:
        return min(self.manifest.budget_per_region - run.hypotheses,
                   self.manifest.budget_total - self.total_hypotheses)

    def _ask_judge(self, run: RegionRun, phase: str, call):
        """One judge call with its log row: (response, None), or (None, why)."""
        t0 = time.perf_counter()
        try:
            resp = call()
        except JudgeBabble:
            self.log.append("judge", fingerprint=run.region.fingerprint, phase=phase,
                            latency_s=round(time.perf_counter() - t0, 1), action="babble")
            return None, "babble"
        except Exception as e:
            # a transport blip (CLI exit, timeout, network) costs a call or a
            # region, never the job
            self.log.append("judge", fingerprint=run.region.fingerprint, phase=phase,
                            latency_s=round(time.perf_counter() - t0, 1),
                            action="error", reason=str(e)[:200])
            return None, f"{type(e).__name__}: {str(e)[:120]}"
        summary = ({"queue": len(resp.queue)} if phase == "seed" else
                   {"mutations": len(resp.mutations),
                    "action": "yield" if resp.kernel is None else "proposal"})
        self.log.append("judge", fingerprint=run.region.fingerprint, phase=phase,
                        latency_s=round(time.perf_counter() - t0, 1), **summary)
        return resp, None

    def _set_head(self, run: RegionRun, kernel: KernelSpec, clock, tag: str) -> None:
        """clock is a ladder result or a recorded attempt: anything with the
        kernel's region_ms and library_ms."""
        get = clock.get if isinstance(clock, dict) else lambda k: getattr(clock, k)
        run.head, run.head_tag = kernel, tag
        run.head_ms, run.library_ms = get("region_ms"), get("library_ms")
        run.head_floor_ms = get("floor_ms")
        run.head_ratio = _ratio(clock)

    def _apply_verdict(self, run: RegionRun, item, kernel: KernelSpec, result) -> str:
        """The spec's three verdicts, plus the bookkeeping each one moves.
        Comparisons between kernels use each one's ratio to the library it
        was clocked beside, so two clocks taken minutes apart still compare."""
        if result.outcome == "failed":
            return "failed"
        ratio, sigma = _ratio(result), (result.sigma_ms or 0.0) / (result.library_ms or 1.0)
        outcome = result.outcome
        if outcome == "tentative_ship" and run.shipped_ratio is not None and not (
            run.shipped_ratio - ratio > max(0.01 * run.shipped_ratio, 3.0 * sigma)
        ):
            # faster than the library, but the installed kernel is faster still
            result.detail["not_faster_than_shipped"] = {
                "candidate_over_library": ratio, "shipped_over_library": run.shipped_ratio}
            outcome = "correct_slower"
        if outcome == "tentative_ship":
            if not self._bind_and_promote(run, kernel, result):
                return "rolled_back"
            run.shipped, run.shipped_ms, run.shipped_ratio = kernel, result.region_ms, ratio
            run.shipped_floor_ms = result.floor_ms
            run.shipped_tag = item.assoc_tag
            self._set_head(run, kernel, result, item.assoc_tag)
            return "shipped"
        # correct but not a win: head moves only to a better correct kernel
        if run.head_ratio is None or ratio < run.head_ratio:
            self._set_head(run, kernel, result, item.assoc_tag)
        return "correct_slower"

    def _record_attempt(self, run: RegionRun, hyp_id: str, kind: str, text: str,
                        assoc_tag: str | None, kernel, parent: str | None, result,
                        outcome: str | None = None, gate: str | None = None,
                        reason: str | None = None) -> None:
        """One attempt, written three ways: the report row, the run.jsonl
        verdict row, and one line of candidates.log."""
        region = run.region
        if result is None:  # the judge produced nothing to evaluate
            outcome, detail = "failed", {"reason": reason} if reason else {}
            region_ms = library_ms = win_ms = sigma_ms = floor_ms = None
        else:
            outcome = outcome or result.outcome
            gate, detail = result.failed_gate, result.detail
            region_ms, library_ms = result.region_ms, result.library_ms
            win_ms, sigma_ms, floor_ms = result.win_ms, result.sigma_ms, result.floor_ms
        if kernel is not None:
            run.last_kernel = kernel.kernel_id
            run.attempts[kernel.kernel_id] = {
                "hypothesis_id": hyp_id, "verdict": outcome, "failed_gate": gate,
                "region_ms": region_ms, "library_ms": library_ms, "win_ms": win_ms,
                "floor_ms": floor_ms,
            }
        if gate:
            how = f"{outcome} at {gate}: {_short_reason(detail)}"
        elif region_ms is not None and library_ms is not None:
            how = (f"{outcome}: {region_ms:.4f} ms vs library {library_ms:.4f} ms per copy "
                   f"(win {win_ms:+.4f}, sigma {sigma_ms:.4f})")
        elif outcome == "correct_slower":
            how = "correct, not clocked"
        else:
            how = outcome
        self.report.add_hypothesis(
            hypothesis_id=hyp_id, region=region.fingerprint, kind=kind,
            hypothesis_text=text, assoc_tag=assoc_tag, parent=parent,
            kernel=kernel.kernel_id if kernel else None, verdict=outcome,
            failed_gate=gate, region_ms=region_ms, library_ms=library_ms,
            win_ms=win_ms, sigma_ms=sigma_ms, floor_ms=floor_ms, summary=how,
        )
        self.log.append("verdict", fingerprint=region.fingerprint, hypothesis=hyp_id,
                        hypothesis_kind=kind, hypothesis_text=text, assoc_tag=assoc_tag,
                        kernel=kernel.kernel_id if kernel else None, parent=parent,
                        outcome=outcome, gate=gate, region_ms=region_ms,
                        library_ms=library_ms, win_ms=win_ms, sigma_ms=sigma_ms,
                        floor_ms=floor_ms, detail=detail)
        self.candidates.append(
            f"{wall_now()}\t{region.fingerprint[:8]}\t{hyp_id}\t{kind}\t{how}\t{text}")

    def _meta(self, run: RegionRun, queue: Queue, writing_for: dict | None) -> dict:
        """Everything the judge gets on one call. The kernels it can name are
        the scaffold, head, shipped, the one the last verdict was about, and
        any a queued item depends on."""
        region = run.region
        io_specs = {}
        for m in region.members:
            if m.workload in io_specs:
                continue
            specs = self.traces[m.workload].span_specs(m.start_seq, m.end_seq)
            io_specs[m.workload] = {
                "inputs": [specs[a] for a in m.input_ids],
                "outputs": [specs[a] for a in m.output_ids],
            }
        for label, span, specs in self._sweep_instances(region):
            io_specs[label] = {
                "inputs": [specs[a] for a in span.input_ids],
                "outputs": [specs[a] for a in span.output_ids],
            }
        rep = region.members[0]
        wanted = {k.kernel_id for k in (run.scaffold, run.head, run.shipped) if k}
        wanted.add(run.last_kernel)
        for item in queue.snapshot():
            if item["depends_on"]:
                wanted.add(f"r{region.fingerprint[:6]}_{item['depends_on']}")
        kernels = {}
        for kid in sorted(wanted - {None}):
            spec = run.kernels.get(kid)
            if spec is not None:
                kernels[kid] = {**_kernel_view(spec), **run.attempts.get(kid, {})}
        chip = {"gpu_cores": gpu_core_count()}
        if self.peaks is not None:
            chip.update(bandwidth_gbps=self.peaks.bandwidth_gbps, launch_us=self.peaks.launch_us,
                        flops_gflops=self.peaks.flops_gflops)
        history = [{k: h[k] for k in ("id", "kind", "hypothesis", "parent", "verdict", "summary")}
                   for h in self.report.hypotheses if h["region"] == region.fingerprint]
        done = [{"ops": r["ops"], "copies": r["copies"], "attempts": r["hypotheses"],
                 "shipped": f"{r['s']:.2f}x the library" if r.get("s") else "nothing",
                 "close": r["close_rule"]} for r in self.report.regions]
        return render_region_state(
            region=region, io_specs=io_specs, chip=chip,
            ops=_ops_view(self.traces[rep.workload], rep),
            kernels=kernels,
            head=run.head.kernel_id if run.head else None,
            shipped=run.shipped.kernel_id if run.shipped else None,
            head_ms=run.head_ms, shipped_ms=run.shipped_ms, assoc_tag=run.head_tag,
            head_floor_ms=run.head_floor_ms, shipped_floor_ms=run.shipped_floor_ms,
            queue=queue, history=history, lessons=list(self.lessons), regions_done=done,
            budget={"attempts_left_region": self.manifest.budget_per_region - run.hypotheses,
                    "attempts_left_job": self.manifest.budget_total - self.total_hypotheses},
            last_verdict=run.attempts.get(run.last_kernel) if run.last_kernel else None,
            writing_for=writing_for,
        )

    def _close_rule(self, run: RegionRun) -> str | None:
        """A region closes when its budget or the job's is spent, and for no
        other reason: the judge is asked until then, whatever the verdicts."""
        if run.hypotheses >= self.manifest.budget_per_region:
            return "the region's hypothesis budget is spent"
        if self.total_hypotheses >= self.manifest.budget_total:
            return "the job budget is spent"
        return None

    # -- bind and promote -----------------------------------------------------

    def _bind_and_promote(self, run: RegionRun, kernel: KernelSpec, result) -> bool:
        region = run.region
        installed_now: list[tuple[str, object]] = []
        emitted_before: dict[str, object] = {}   # scope -> prior entry (None = absent)
        pending_installed: dict[str, tuple] = {}
        try:
            self.tracer.patcher.install()
        except RuntimeError:
            pass  # already installed

        def unwind():
            """Every failure path must leave the model, the artifact record,
            and the patch surface exactly as before this attempt."""
            for path, occupant in reversed(installed_now):
                try:
                    swap_uninstall(self.model, path, occupant)
                except Exception as e:
                    self.log.append("rollback_error", scope=path, reason=str(e))
            for path, prior in emitted_before.items():
                if prior is None:
                    self.emitted.pop(path, None)
                else:
                    self.emitted[path] = prior
            try:
                self.tracer.uninstall()
            except Exception:
                pass

        baseline_traces = dict(self.traces)
        try:
            for m in region.members:
                trace = self.traces[m.workload]
                scope = find_scope_call(trace, m.scope_stack)
                if scope is None:
                    raise NotReplayable(f"no scope call for {m.scope_stack!r}")
                scope_path = scope.address.rsplit("@", 1)[0]
                if scope_path == "":
                    raise NotReplayable("root-scope delivery is not supported yet")

                if scope_path not in self.certified_scopes:
                    emitted_id = emit_wrapper(trace, scope, [], f"Id_{_safe(scope_path)}")
                    cls = _load_class(emitted_id)
                    original = _resolve(self.model, scope_path)
                    check = certify_identity(
                        build_wrapper=lambda c=cls, o=original: c(o, {}),
                        install=lambda w, p=scope_path: self._swap_in(p, w),
                        runs=[lambda t=self.tensors[w.name]: self.model(*t)
                              for w in self.manifest.workloads],
                    )
                    if not check.ok:
                        self.log.append("certification_failed", scope=scope_path, reason=check.reason)
                        unwind()
                        return False
                    self.certified_scopes.add(scope_path)

                splice = Splice(
                    kernel=kernel, start_seq=m.start_seq, end_seq=m.end_seq,
                    input_ids=m.input_ids, output_ids=m.output_ids,
                    fingerprint=region.fingerprint,
                )
                # one wrapper per scope carries every cut shipped into it, the
                # copies of this attempt included; a cut on a span already
                # spliced replaces the old kernel there
                prior = pending_installed.get(scope_path) or self.installed.get(scope_path)
                if prior:
                    original, prior_splices, _ = prior
                    splices = [s for s in prior_splices
                               if (s.start_seq, s.end_seq) != (m.start_seq, m.end_seq)] + [splice]
                else:
                    original = _resolve(self.model, scope_path)
                    splices = [splice]
                kernels = {s.kernel.kernel_id: s.kernel for s in splices}
                emitted = emit_wrapper(trace, scope, splices, f"W_{_safe(scope_path)}")
                cls = _load_class(emitted)
                occupant = swap_install(self.model, scope_path, cls(original, kernels))
                installed_now.append((scope_path, occupant))
                emitted_before.setdefault(scope_path, self.emitted.get(scope_path))
                self.emitted[scope_path] = emitted
                pending_installed[scope_path] = (original, splices, kernels)

            # retrace: every installed cut, this one included, must be exactly
            # one custom dispatch per copy against the job-start recording
            pending_cuts: dict[str, dict[tuple[int, int], str]] = {}
            for w in self.manifest.workloads:
                new = {(m.start_seq, m.end_seq): kernel.kernel_id
                       for m in region.members if m.workload == w.name}
                if not new:
                    continue
                cuts = {**self.cuts.get(w.name, {}), **new}
                spans = sorted(cuts)
                retrace, _ = self.tracer.trace(self.model, self.tensors[w.name])
                rep_check = verify_retrace(baseline_traces[w.name], retrace, spans,
                                           [cuts[s] for s in spans])
                if not rep_check.ok:
                    raise NotReplayable("; ".join(rep_check.reasons))
                pending_cuts[w.name] = cuts

            self.tracer.uninstall()
            e2e = run_e2e(
                self.session, self.baseline_model, self.model,
                workloads=[(w.name, self.tensors[w.name]) for w in self.manifest.workloads],
                timed=self._timed_arms(),
                veto_pairs=8,
            )
            if not e2e.passed:
                veto = e2e.veto
                self.log.append("e2e_failed", fingerprint=region.fingerprint,
                                checks=[c.__dict__ for c in e2e.checks],
                                veto_passed=e2e.veto_passed,
                                veto={"baseline_ms": veto.median_baseline_ms,
                                      "delta_ms": veto.median_delta_ms, "sigma_ms": veto.sigma_ms,
                                      "ratio": veto.median_ratio, "stability": veto.stability}
                                if veto else None)
                unwind()
                return False
            self.installed.update(pending_installed)
            self.cuts.update(pending_cuts)
            self.log.append("shipped", fingerprint=region.fingerprint, kernel=kernel.kernel_id)
            return True
        except Exception as e:
            # NotReplayable is the expected member; anything else is equally
            # rolled back so the model is never left half-patched
            unwind()
            self.log.append("bind_failed", fingerprint=region.fingerprint,
                            reason=f"{type(e).__name__}: {e}")
            return False

    def _swap_in(self, path: str, wrapper):
        occupant = swap_install(self.model, path, wrapper)
        return lambda: swap_uninstall(self.model, path, occupant)

    # -- the whole job --------------------------------------------------------

    def run(self) -> Report:
        self.log.append(
            "job", manifest=self.report.manifest_path,
            workloads={w.name: [[str(d) for d in i.shape] for i in w.inputs]
                       for w in self.manifest.workloads},
            budget_per_region=self.manifest.budget_per_region,
            budget_total=self.manifest.budget_total,
            defaulted=list(self.manifest.defaulted),
        )
        self.load_model()
        self.trace_workloads()
        regions = self.build_regions()
        ranked = self.capture_and_price(regions)
        self.report.write(self.work_dir / "report.json")  # partial: peaks, coverage, strands

        shipped_regions: list[Region] = []
        for region in ranked:
            free = free_members(region, shipped_regions)
            if not free:
                self.log.append("region_covered", fingerprint=region.fingerprint,
                                reason="every copy touches a cut already shipped")
                continue
            if len(free) < region.copies:
                # a shipped bigger cut owns the other copies; this region goes
                # on at the copies still free, priced at that count
                self.log.append("region_trimmed", fingerprint=region.fingerprint,
                                copies_before=region.copies, copies_free=len(free))
                region.members = free
                for w in region.workloads:
                    n = sum(m.workload == w for m in free)
                    region.t_orig_ms[w] = region.t_rep_ms.get(w, 0.0) * n
                    region.p[w] = region.p_rep.get(w, 0.0) * n
            roof = region.roofline
            self.log.append(
                "region_open", fingerprint=region.fingerprint, ops=list(region.ops),
                copies=region.copies, p=dict(region.p),
                bound=roof.bound if roof else None, s_max=roof.s_max if roof else None,
                roofline_ms=roof.t_roofline_ms if roof else None,
                t_orig_ms=dict(region.t_orig_ms), t_rep_ms=dict(region.t_rep_ms),
            )
            self.candidates.append(_region_line(region, "open"))
            if self.total_hypotheses >= self.manifest.budget_total:
                run = RegionRun(region=region, close_rule="not opened: the job budget is spent")
                self.log.append("region_skip", fingerprint=region.fingerprint, reason=run.close_rule)
            else:
                judge = self.judge_factory(region)
                run = self.open_region(region, judge)
                if run.scaffold is not None and run.close_rule is None:
                    self.hypothesis_cycle(run, judge)
            if run.shipped is not None:
                shipped_regions.append(region)
            self.report.add_region(
                fingerprint=region.fingerprint, ops=list(region.ops),
                copies=region.copies, workloads=list(region.workloads),
                p=dict(region.p), t_orig_ms=dict(region.t_orig_ms),
                t_rep_ms=dict(region.t_rep_ms),
                roofline_ms=roof.t_roofline_ms if roof else None,
                bound=roof.bound if roof else None, s_max=roof.s_max if roof else None,
                t_shipped_ms={w: run.shipped_ms for w in region.workloads}
                    if run.shipped_ms else None,
                speedup=(1.0 / run.shipped_ratio) if run.shipped_ratio else None,
                close_rule=run.close_rule, hypotheses=run.hypotheses, head_ms=run.head_ms,
            )
            self.log.append("region_closed", fingerprint=region.fingerprint,
                            rule=run.close_rule, shipped=run.shipped is not None,
                            hypotheses=run.hypotheses, head_ms=run.head_ms,
                            shipped_ms=run.shipped_ms,
                            outcomes=self.report.regions[-1]["outcomes"])
            self.candidates.append(_region_line(region, "closed", run))
            self.report.write(self.work_dir / "report.json")  # a crash still leaves the story so far

        # The headline is a ratio of two clocks taken together: the
        # untouched model is re-measured here, interleaved with the patched one,
        # never subtracted from the job-start clock taken on a different machine
        # state. "before" stays in the report as the job-start observation it is.
        for w in self.manifest.workloads:
            tensors = self.tensors[w.name]
            comp = compare(
                self.session,
                self._step_fn(self.baseline_model, tensors),
                self._step_fn(self.model, tensors),
                pairs=self.clock_pairs,
            )
            after = statistics.median(comp.candidate_ms)
            self.report.step_ms[w.name].update({
                "after": after,
                "baseline_at_end": comp.median_baseline_ms,
                "speedup": (1.0 / comp.median_ratio) if comp.median_ratio else None,
                "stability": comp.stability,
            })
            self.log.append("step_clock", workload=w.name, phase="after",
                            median_ms=after, baseline_at_end_ms=comp.median_baseline_ms,
                            speedup=self.report.step_ms[w.name]["speedup"],
                            stability=round(comp.stability, 3))

        self._final_check()
        self.report.session["idled_s"] = round(self.session.idled_s, 1)
        self.report.constants = {
            "min_win_ms": MIN_WIN_MS, "budget_per_region": self.manifest.budget_per_region,
            "budget_total": self.manifest.budget_total, "seed": self.manifest.seed,
            "clock_pairs": self.clock_pairs,
            "defaulted": list(self.manifest.defaulted),
        }
        self.report.write(self.work_dir / "report.json")
        return self.report

    def _final_check(self) -> None:
        """The last end-to-end check once the regions are done: the patched
        model as a whole against the untouched model, on every workload and
        then at every sweep size, where the wrappers must hand the unrecorded
        shapes back to the original modules."""
        if not self.installed:
            return
        final = run_e2e(
            self.session, self.baseline_model, self.model,
            workloads=[(w.name, self.tensors[w.name]) for w in self.manifest.workloads]
            + sorted(self.sweep_tensors.items()),
            timed=self._timed_arms(),
            veto_pairs=8,
        )
        self.final_ok = final.passed
        self.report.final = {"passed": final.passed, "veto_passed": final.veto_passed,
                             "checks": [c.__dict__ for c in final.checks]}
        self.log.append("final_e2e", passed=final.passed, veto_passed=final.veto_passed,
                        checks=self.report.final["checks"])

    def emit_artifact(self, out_dir: str | Path) -> Path:
        """Write the artifact, then load it onto a fresh build() in a fresh
        process and check it reproduces the patched model."""
        if not self.final_ok:
            raise RuntimeError("the final whole-model check failed; no artifact is written "
                               "for a patched model that does not match the original")
        specs: dict[str, KernelSpec] = {}
        for path in self.emitted:
            wrapper = _resolve(self.model, path)
            specs.update(getattr(wrapper, "_specs", {}))
        out = emit_artifact(out_dir, list(specs.values()), list(self.emitted.values()), self.report)
        if self.installed:
            first = self.manifest.workloads[0].name
            tensors = self.tensors[first]
            check_apply(out, self.manifest.model_path, tensors, flatten_arrays(self.model(*tensors)))
            self.log.append("artifact_checked", artifact=str(out), workload=first)
        return out


def _safe(path: str) -> str:
    return path.replace(".", "_") or "root"


def _ops_view(trace: Trace, stretch: Stretch) -> list[dict]:
    """The recorded calls of one copy, with arrays named by role (in0, out0,
    t<seq> for intermediates) and every non-tensor argument spelled out."""
    names = {aid: f"in{i}" for i, aid in enumerate(stretch.input_ids)}
    names.update({aid: f"out{i}" for i, aid in enumerate(stretch.output_ids)})
    view = []
    for node in trace.nodes[stretch.start_seq:stretch.end_seq + 1]:
        for aid in node.out_arrays:
            names.setdefault(aid, f"t{node.seq}")

        def name(obj, node=node):
            if isinstance(obj, ArrayRef):
                return names[node.in_arrays[obj.index]]
            if isinstance(obj, (list, tuple)):
                return [name(v) for v in obj]
            if isinstance(obj, dict):
                return {k: name(v) for k, v in obj.items()}
            if isinstance(obj, slice):
                return f"slice({obj.start}, {obj.stop}, {obj.step})"
            return obj if isinstance(obj, (int, float, bool, str)) or obj is None else str(obj)

        view.append({
            "op": node.op,
            "args": name(list(node.scalar_args["args"])),
            "kwargs": name(dict(node.scalar_args["kwargs"])),
            "outputs": [names[aid] for aid in node.out_arrays],
        })
    return view


def _ratio(clock) -> float | None:
    """A kernel's time over the library's, both from the same clock."""
    get = clock.get if isinstance(clock, dict) else lambda k: getattr(clock, k)
    region_ms, library_ms = get("region_ms"), get("library_ms")
    return (region_ms / library_ms) if region_ms and library_ms else None


def _item_view(item) -> dict:
    return {"id": item.id, "kind": item.kind, "assoc_tag": item.assoc_tag,
            "hypothesis": item.hypothesis}


def _verdict_payload(hyp_id: str, kernel_id: str, outcome: str, result) -> dict:
    return {
        "hypothesis_id": hyp_id, "kernel_id": kernel_id, "outcome": outcome,
        "failed_gate": result.failed_gate, "detail": _safe_detail(result.detail),
        "region_ms": result.region_ms, "library_ms": result.library_ms,
        "win_ms": result.win_ms, "sigma_ms": result.sigma_ms, "floor_ms": result.floor_ms,
    }


def _short_reason(detail: dict | None) -> str:
    """The first thing worth reading in a failed gate's detail, on one line."""
    if not detail:
        return ""
    if detail.get("failures"):
        f = detail["failures"][0]
        return f"{f.get('check')}: {f.get('detail')}"
    if detail.get("diagnostics"):
        d = detail["diagnostics"][0]
        return f"line {d.get('body_line')}: {d.get('message')}"
    parts = [str(detail[k]) for k in ("regime", "kind", "reason", "note") if detail.get(k)]
    return "; ".join(parts)[:160]


def _region_line(region: Region, event: str, run: "RegionRun | None" = None) -> str:
    roof = region.roofline
    rep = ", ".join(f"{w}={ms:.4f}" for w, ms in region.t_rep_ms.items())
    if event == "open":
        how = f"copies={region.copies}\tlibrary ms per copy: {rep}"
        if roof:
            how += f"\troofline ms per copy: {roof.t_roofline_ms:.4f}"
    else:
        shipped = run.shipped.kernel_id if run and run.shipped else "none"
        how = f"{run.close_rule}\thypotheses={run.hypotheses}\tshipped={shipped}"
    return f"{wall_now()}\t{region.fingerprint[:8]}\tregion_{event}\t{' > '.join(region.ops)}\t{how}"


HEADER_SHOWN_CHARS = 8000  # a judge-written header fits; a library header does not


def resolve_parent(run: RegionRun, name: str) -> KernelSpec | None:
    """The kernel a proposal edits: head, scaffold, shipped, a hypothesis id,
    or a kernel id; None when the name matches nothing in this region."""
    by_role = {"head": run.head, "scaffold": run.scaffold, "shipped": run.shipped}
    if name in by_role:
        return by_role[name]
    if name in run.kernels:
        return run.kernels[name]
    return run.kernels.get(f"r{run.region.fingerprint[:6]}_{name}")


def kernel_from_proposal(contract: RegionContract, parent: KernelSpec | None, proposal,
                         kernel_id: str) -> KernelSpec:
    """The harness owns names, dtypes, and the call site; the judge supplies
    the body and the launch. A header or template left out means the parent's."""
    scratch = tuple(proposal.scratch)
    return KernelSpec(
        kernel_id=kernel_id,
        name=f"at_{kernel_id}",
        input_names=contract.input_names,
        output_names=contract.output_names + tuple(s[0] for s in scratch),
        source=proposal.source,
        header=proposal.header or (parent.header if parent else ""),
        grid=tuple(proposal.grid),
        threadgroup=tuple(proposal.threadgroup),
        output_shapes=(tuple(tuple(s) for s in proposal.output_shapes)
                       + tuple(tuple(s[2]) for s in scratch)),
        output_dtypes=contract.output_dtypes + tuple(s[1] for s in scratch),
        template=tuple(proposal.template) or (parent.template if parent else ()),
        fallback_predicate=proposal.fallback_predicate,
    )


def _kernel_view(spec: KernelSpec) -> dict:
    """What the judge may see of a kernel: its source and launch story.
    Never the call-site fields the harness owns (init_value, math_mode)."""
    header = spec.header
    if len(header) > HEADER_SHOWN_CHARS:
        header = (f"<{header.count(chr(10))} lines of the library's own Metal source, kept by "
                  f"the harness; leave header out of a proposal to keep it>")
    return {
        "source": spec.source,
        "header": header,
        "grid": list(spec.grid),
        "threadgroup": list(spec.threadgroup),
        "output_shapes": [list(s) for s in spec.output_shapes],
        "output_dtypes": list(spec.output_dtypes),
        "template": [list(t) for t in spec.template],
        "input_names": list(spec.input_names),
        "output_names": list(spec.output_names),
    }


# the acceptance envelope: values the judge could tune an edit to sit under
_ENVELOPE_KEYS = frozenset({"rtol", "atol", "tolerance", "kappa", "floor",
                            "changing_floor", "err_library", "err_candidate"})


def _safe_detail(detail: dict | None) -> dict:
    """Judge-visible gate detail: distances like max_excess only, never the
    acceptance envelope. The judge may learn how far a check missed, never
    where the line is."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in _ENVELOPE_KEYS}
        if isinstance(value, (list, tuple)):
            return [strip(v) for v in value]
        return value
    return strip(detail) if detail else {}


def _load_class(emitted: EmittedWrapper):
    from .bind.emit import MODULE_HEADER

    ns: dict = {}
    exec(compile(MODULE_HEADER + emitted.source, f"<wrapper {emitted.class_name}>", "exec"), ns)
    return ns[emitted.class_name]


def _resolve(model, path: str):
    from autotuner_runtime.swap import resolve

    _, _, child = resolve(model, path)
    return child


def run_job(manifest_path, work_dir, judge_factory, session=None) -> Report:
    return JobRunner(manifest_path, work_dir, judge_factory, session=session).run()
