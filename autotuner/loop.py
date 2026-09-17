"""The job driver and region loop (spec "The loop" and "Closing a region").

One hypothesis at a time. The harness owns measurement, correctness, bind, and
the artifact; the judge proposes. Verdicts: failed, correct-but-slower (climb),
tentative ship, then bind and e2e decide whether a ship is real.
"""

from __future__ import annotations

from autotuner_runtime.numeric import DEFAULT_TOLERANCES

import importlib.util
import copy
import statistics
import sys
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import mlx.core as mx

from . import manifest as manifest_mod
from .bind.certify import certify_identities, find_scope_call, screen_scope
from .bind.emit import (EmittedWrapper, NotReplayable, Splice, compose_scope_variants,
                        emit_wrapper, emit_wrapper_variants)
from .bind.swap import install as swap_install, uninstall as swap_uninstall
from .bind.verify import verify_retrace
from .e2e import E2EResult, _flatten_params, preserving_check, run_e2e, share_weights
from .judge.directions import OPENERS, directions_for
from .judge.prompts import ENVELOPE_KEYS, render_region_state
from .judge.queue import Queue, QueueError
from .judge.schema import JudgeBabble
from .ladder.gates import EvalSet, LadderJob, LadderResult, run_ladder
from .ladder.static_checks import RegionContract, SINGLE_GROUP_ELEMENT_LIMIT
from .artifact.bundle import ModelBundle
from .artifact.emit import check_apply_many, emit_artifact, write_kernel
from .log import RunLog, TextLog, wall_now
from .measure.clocks import compare, comparison_from_samples, step_clock
from autotuner_runtime.graph import GraphWrapper
from .measure.controls import aa_null
from .measure.peaks import (BUSY_GPU_PERCENT, gpu_core_count, gpu_utilization,
                            implausible as peaks_implausible, measure_peaks)
from .measure.session import Session
from .measure.sequences import compare_sequences
from .regions.build import build_stretches, is_view, weight_like_ids
from .regions.fingerprint import group_copies
from .regions.price import PRICE_PAIRS, capture_boundaries, capture_instances, price_group
from .regions.rank import (REGION_FLOOR_P, apply_floor, estimate_regions, free_members,
                           rank, select_frontier)
from .regions.roofline import observed_peaks, step_floor, stretch_roofline
from .regions.store import BoundaryStore
from .regions.types import Region, Stretch
from .report import Report
from .sandbox.watchdog import GPU_WINDOW_S
from .scaffold import uncovered_op
from .scaffold.lower import SINGLE_GROUP_WORK_LIMIT
from .trace import Tracer
from .trace.recorder import ArrayRef
from .trace.walk import flatten_arrays
from .regions.sweep import SweepDivergence, locate_span
from .trace.serialize import nodes_to_json
from .trace.types import Trace
from .workload import context_tokens, materialize, workload_seeds
from autotuner_runtime.kernels import KernelSpec
from autotuner_runtime.exact import bitwise_equal
from autotuner_runtime.swap import require_independent_models
from autotuner_runtime.sequence import make_sequence
from autotuner_runtime.state import ContextSequence, context_step
from autotuner_runtime.stats import workload_win

MIN_WIN_MS = 0.030  # a win must save a few tens of microseconds per step across copies
CLOCK_PAIRS = 32    # ABBA pairs behind the ship clock and the headline
LESSONS_KEPT = 24   # the judge's newest lessons travel with every call
LESSON_CONTEXT_CHARS = 400  # full notes remain in the log; prompts carry excerpts
# Each comparison has ten ABBA blocks (forty forward calls). Resolved wins
# receive one fresh comparison before installation; inconclusive results stop.
SHIP_PAIRS = 20
@dataclass
class PromotionResult:
    status: str
    reason: str = ""
    checks: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)

    def __bool__(self):
        return self.status == "shipped"

    @property
    def outcome(self):
        if self:
            return "shipped"
        return "rolled_back" if self.status in {"binding_failed", "correctness_failed"} else "correct_slower"


def _promotion_feedback(result, promotion):
    """Keep local nomination separate from the whole-model decision."""
    if "ship" in result.detail:
        result.detail["region_nominated"] = result.detail.pop("ship")
    result.detail["model_check"] = asdict(promotion)


@dataclass
class RegionRun:
    region: Region
    scaffold: KernelSpec | None = None
    head: KernelSpec | None = None
    shipped: KernelSpec | None = None
    head_ms: float | None = None
    shipped_ms: float | None = None
    head_ratio: float | None = None   # head_ms over the library beside it: the drift-free number
    shipped_ratio: float | None = None
    head_floor_ms: float | None = None     # the floor probe clocked beside head, same child
    shipped_floor_ms: float | None = None
    head_sigma_ms: float | None = None     # uncertainty of head's win, same clock
    head_age: int = 0                      # evaluated kernels since head last moved
    floor_streak: int = 0                  # consecutive kernels within one sigma of the floor beside them
    directions: list[dict] = field(default_factory=list)  # what the widening round opens from
    openers: list[str] = field(default_factory=list)      # kinds the widening round opened
    head_tag: str = "preserving"      # assoc tag of the edit that produced head
    head_workload: str | None = None
    shipped_workload: str | None = None
    last_kernel: str | None = None    # the kernel the latest verdict was about
    attempts: dict[str, dict] = field(default_factory=dict)  # kernel id -> its verdict
    hypotheses: int = 0
    errors: int = 0         # consecutive transport failures; three stop the job
    refused: int = 0        # consecutive replies with nothing to evaluate
    empty_replies: int = 0  # all such replies, to name each one once
    close_rule: str | None = None
    kernels: dict[str, KernelSpec] = field(default_factory=dict)
    queue: Queue | None = None
    last_verdict: dict | None = None


class JobRunner:
    use_library_inference: bool | None = None  # resolved against build() before tracing

    def __init__(self, manifest_path: str | Path, work_dir: str | Path,
                 judge_factory, session: Session | None = None,
                 clock_pairs: int = CLOCK_PAIRS):
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
        self.baseline = self.manifest.baseline  # settled by _clock_steps once the traces are in
        # the workload whose call is a step over a filled KV cache, set by load_model
        self.context = next((w for w in self.manifest.workloads if w.context is not None), None)
        self.context_seed = self.context_tokens = None
        self.session = session or Session(log_path=self.work_dir / "session.jsonl")
        self.log = RunLog(self.work_dir / "run.jsonl")
        self.candidates = TextLog(self.work_dir / "candidates.log")
        self.kernel_dir = self.work_dir / "kernels"
        self.report = Report(manifest_path=str(manifest_path))
        self.report.constants = {
            "min_win_ms": MIN_WIN_MS, "budget_per_region": self.manifest.budget_per_region,
            "budget_total": self.manifest.budget_total, "seed": self.manifest.seed,
            "openers_per_region": OPENERS,
            "clock_pairs": self.clock_pairs, "price_pairs": PRICE_PAIRS,
            "ship_pairs": SHIP_PAIRS, "model_min_gain_pct": 0.0,
            "model_confidence_sigma": 3.0, "model_confirmation_comparisons": 1,
            "final_benchmark": asdict(self.manifest.final_benchmark),
            "numeric_policy": "exact_preserving_baseline_tolerance_changing",
            "tolerances": self.manifest.tolerances,
            "tolerance_defaults": dict(DEFAULT_TOLERANCES),
            "duty_idle_factor": self.session.duty_idle_factor,
            "gpu_evaluation_timeout_s": GPU_WINDOW_S,
            "single_group_element_limit": SINGLE_GROUP_ELEMENT_LIMIT,
            "single_group_scaffold_work_limit": SINGLE_GROUP_WORK_LIMIT,
            "defaulted": list(self.manifest.defaulted),
        }
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
        self.peaks = None  # measured by measure_machine, after the step clock
        self.gpu_busy_at_start = gpu_utilization()  # the counter trails: read before any GPU work of ours
        self.total_hypotheses = 0
        self.lessons: list[dict] = []  # what the judge wrote down for later regions
        # scope -> (original module, {(workload, call address): splices}, kernels)
        self.installed: dict[str, tuple] = {}
        self.scaffold_overrides: dict[str, KernelSpec] = {}
        self.cuts: dict[str, dict[tuple[int, int], str]] = {}  # workload -> {span: kernel id}
        self.certified_scopes: set[str] = set()
        self._compiled_baseline = None  # under a compiled baseline: the untouched model with its baseline scopes compiled
        self.replay_scopes: set[str] = set()
        self.emitted: dict[str, EmittedWrapper] = {}
        self.final_ok = False  # export requires a completed final check
        # Historical product of accepted step ratios, for diagnostics only.
        # Ship decisions remeasure both candidate and installed incumbent.
        self.model_ratio = 1.0
        self.model_ratios = {w.name: 1.0 for w in self.manifest.workloads}
        self.pending_regions: list[Region] = []
        self.selection_wave = 0
        self.shipped_tags: dict[str, str] = {}

    # -- stage 1: model and traces -------------------------------------------

    def load_model(self):
        from autotuner.artifact.bundle import (check_model_dependencies, capture_checkpoint_loads,
                                               resolve_model_checkpoints)
        check_model_dependencies(self.manifest.model_path)
        from autotuner_runtime.graph_native import prepare
        prepare()  # build/ABI failures must happen before model search
        manifest_mod.check_build(self.manifest)
        self.tracer.install(model_module_name="autotune_model")
        try:
            spec = importlib.util.spec_from_file_location("autotune_model", self.manifest.model_path)
            module = importlib.util.module_from_spec(spec)
            with capture_checkpoint_loads() as captured:
                spec.loader.exec_module(module)
                self.model_module = module
                self.build_model = self._build_model
                self.model = self.build_model()
                self.baseline_model = self.build_model()
                require_independent_models(self.model, self.baseline_model)
            self.checkpoint_pins = resolve_model_checkpoints(self.manifest.model_path, captured=captured)
            self.report.constants["checkpoint_pins"] = self.checkpoint_pins
        except BaseException:
            self.tracer.uninstall()  # a failed load leaves no patches behind
            raise
        shared = share_weights(self.model, self.baseline_model)
        mx.clear_cache()  # return the freed second weight copy to the OS
        total = len(_flatten_params(self.baseline_model.parameters())) \
            if hasattr(self.baseline_model, "parameters") else 0
        context = None if self.context is None else {
            "workload": self.context.name, "tokens": self.context.context, "seed": self.context_seed}
        self.log.append("model", shared_weights=shared, parameters=total, context=context)
        if shared < total:
            self._env_warning(f"only {shared} of {total} parameters could be shared between "
                              "the two model copies; both stay resident, which adds noise "
                              "to the whole-model checks")

    def _build_model(self):
        """Every build retains custom definitions, even between trace passes."""
        from autotuner_runtime.captured_kernels import capture_construction
        from .artifact.bundle import use_checkpoint_pins
        with capture_construction(), use_checkpoint_pins(getattr(self, "checkpoint_pins", None)):
            return self._as_step(self.model_module.build())

    def _as_step(self, model):
        """The model as the job runs it: as built, or, when the workload
        declares a context, as one repeatable step over the model's own KV
        cache filled with that many seeded tokens."""
        from autotuner_runtime.inference import LibraryInference, resolve_library_inference
        selected = resolve_library_inference(
            model, self.manifest.use_library_inference, self.manifest.workloads,
            dims=self.manifest.primary, sweep=self.manifest.sweep)
        if self.use_library_inference is not None and selected != self.use_library_inference:
            raise manifest_mod.ManifestError("build() changed its inference compatibility between builds")
        self.use_library_inference = selected
        self.report.constants["use_library_inference"] = selected
        self.report.constants["measurement"] = {
            "kind": "library_generation" if selected else "forward",
            "generated_tokens": self.manifest.final_benchmark.steps if selected else None,
            "baseline": "unmodified library inference" if selected else self.manifest.baseline,
        }
        w = self.context
        if selected:
            if w is not None:
                self.context_seed = workload_seeds(self.manifest.seed, f"{w.name}:context", 1)[0]
                self.context_tokens = context_tokens(w, self.manifest.primary, self.context_seed)
            return LibraryInference(model, steps=self.manifest.final_benchmark.steps,
                                    prefix_tokens=self.context_tokens)
        if w is None:
            return model
        self.context_seed = workload_seeds(self.manifest.seed, f"{w.name}:context", 1)[0]
        self.context_tokens = context_tokens(w, self.manifest.primary, self.context_seed)
        warm = materialize(w, self.manifest.primary, self.context_seed)
        try:
            return context_step(model, w.context, self.context_tokens, warm)
        except TypeError as e:
            raise manifest_mod.ManifestError(f"workload {w.name!r} declares a context, but {e}") from None

    def trace_workloads(self):
        for w in self.manifest.workloads:
            seeds = workload_seeds(self.manifest.seed, w.name, manifest_mod.BOUNDARY_INPUT_SETS)
            self.tensors[w.name] = materialize(w, self.manifest.primary, seeds[0])
            trace, _ = self.tracer.trace(self.model, self.tensors[w.name])
            self.traces[w.name] = trace
            if trace.in_pass_evaluation:
                self.log.append("memory_warning", workload=w.name,
                                detail="model evaluated mid-record; intermediates stayed resident")
            self.log.append("trace", workload=w.name, nodes=len(trace.nodes), never_evaluated=len(trace.dead))
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
            if any(m.scope_stack[-1].rsplit("@", 1)[0] == "" for m in r.members):
                # Installation swaps a module; the model's own top-level call
                # has no parent to swap under, so no attempt could ever land.
                reason = "the region runs at the model's top level, where no module can be swapped"
            if reason is not None:
                r.rejected = f"no certified delivery scope: {reason}"
            elif "metal_kernel" not in r.ops and (op := uncovered_op(r.ops)) is not None:
                # nothing to edit if the harness cannot write a starting kernel;
                # say so before any capture or pricing is spent on it
                r.rejected = f"no scaffold for {op}"
            else:
                self._plan_delivery(r)
                try:
                    self._build_scaffold(r)
                except Exception as error:
                    # Source generation is CPU-only. Discover an unsupported
                    # shape or argument before capturing and timing the cut.
                    r.rejected = f"cannot build scaffold: {type(error).__name__}: {error}"
            if r.rejected:
                self.report.stranded.append({"fingerprint": r.fingerprint, "ops": list(r.ops),
                                             "reason": r.rejected})
            else:
                viable.append(r)
        self.log.append("regions", candidates=len(regions), screened=len(viable))
        self.report.coverage["discovery"] = {
            "legal_regions": len(regions), "supported_regions": len(viable),
            "unsupported_regions": len(regions) - len(viable),
            "never_evaluated_ops": sum(len(t.dead) for t in self.traces.values())}
        return viable

    def _plan_delivery(self, region: Region) -> None:
        """How a kernel will install at the region's scope, read from the
        record before any clock: the kernel call alone when the region is the
        whole scope, graph insertion where compiling the scope keeps its Python
        behavior, else replay. Every clock's library arm follows this choice,
        so a kernel is measured against what its scope will actually run."""
        from .bind.emit import covers_scope, scope_nodes
        from .bind.graph import graph_scope_reason
        for member in region.members:
            if member.workload in region.delivery:
                continue
            trace = self.traces[member.workload]
            scope = find_scope_call(trace, member.scope_stack)
            path = scope.address.rsplit("@", 1)[0] if scope else ""
            if path == "":
                region.delivery[member.workload] = "replay"
                continue
            # One wrapper serves every recorded call of the scope, so every
            # call decides: a cut must be the whole calculation in each, and
            # a compiled scope must keep its Python in each.
            calls = [sc for sc in trace.scope_calls if sc.address.rsplit("@", 1)[0] == path]
            spans = {(m.start_seq, m.end_seq) for m in region.members if m.workload == member.workload}
            if all(covers_scope(nodes, {span for span in spans
                                        if nodes[0].seq <= span[0] and span[1] <= nodes[-1].seq})
                   for sc in calls if (nodes := scope_nodes(trace, sc))):
                region.delivery[member.workload] = "direct"
                continue
            reason = next((r for sc in calls if (r := graph_scope_reason(trace, sc))), None)
            region.delivery[member.workload] = "replay" if reason else "graph"
            if reason:
                region.delivery_reasons[member.workload] = reason

    def capture_and_price(self, regions: list[Region]) -> list[Region]:
        """Clock the step and chip, then capture and price distinct regions.

        The step is clocked before the peaks, not after. measure_machine is
        the heaviest burst of GPU work in the job and pays it back as the
        longest pacing idle in the job; a step clocked right after that idle
        read 8-16 ms for a model that runs at 5.7, because idling slows the
        CPU side and this step spends ~40% of its time building ops in Python.
        Clocking first costs nothing: the room line needs both numbers and is
        computed after either way."""
        # the recorder must be fully removed before any timing
        self.tracer.uninstall()
        mx.clear_cache()  # capture's activation buffers; warm-up refills what clocks need
        self._clock_steps()
        self._settle_delivery(regions)
        self.measure_machine()
        self._record_step_floor()
        self.pending_regions = list(regions)
        self.report.coverage["selection"] = {"candidates": len(regions), "waves": []}
        return self._next_regions()

    def _next_regions(self, blockers: list[Region] = ()) -> list[Region]:
        """Price one set of alternatives. Overlapping cuts keep their turn."""
        while self.pending_regions and self.total_hypotheses < self.manifest.budget_total:
            eligible = [r for r in self.pending_regions
                        if len(free_members(r, blockers)) == r.copies]
            estimates = estimate_regions(eligible, self.traces, self.peaks, self.step_ms)
            ready, deferred = select_frontier(eligible, estimates)
            if not ready:
                return []
            chosen = {r.fingerprint for r in ready}
            self.pending_regions = [r for r in self.pending_regions if r.fingerprint not in chosen]
            self.selection_wave += 1
            row = {"wave": self.selection_wave, "selected": len(ready),
                   "deferred": len(self.pending_regions), "conflicts": deferred}
            self.report.coverage["selection"]["waves"].append(row)
            self.log.append("selection", **row)
            print(f"selection {self.selection_wave}: measuring {len(ready)} regions; "
                  f"{len(self.pending_regions)} overlapping alternatives queued", flush=True)
            self.report.write(self.work_dir / "report.json")
            self.tracer.install()
            try:
                self._capture(ready)
            finally:
                self.tracer.uninstall()
            ranked = self._price_and_rank(ready)
            self.report.write(self.work_dir / "report.json")
            # Already-priced candidates should get their search turn after
            # one alternative wave, even if that wave falls below the floor.
            # Without this boundary a long chain of rejected alternatives
            # sends us back through exhaustive pricing while useful work waits.
            if ranked or blockers:
                return ranked
        return []

    def _refresh_after_ship(self, ranked: list[Region], shipped: list[Region]) -> list[Region]:
        """Remove occupied copies and refresh any remaining prices safely."""
        recapture = []
        for pool in (ranked, self.pending_regions):
            for candidate in list(pool):
                remaining = free_members(candidate, shipped)
                if not remaining:
                    pool.remove(candidate)
                    self.log.append("region_covered", fingerprint=candidate.fingerprint,
                                    reason="every copy touches a shipped cut")
                elif len(remaining) != candidate.copies:
                    candidate.members = remaining
                    if pool is ranked:
                        # The old representative may have been removed. The
                        # same workload label can now name different array IDs
                        # or shapes; old boundary files cannot price that copy.
                        recapture.append(candidate)
        if recapture:
            self.tracer.install()
            try:
                self._capture(recapture)
            finally:
                self.tracer.uninstall()
        # These clocks predate the installed change. Pending regions are
        # captured and priced by _next_regions when their turn arrives.
        return self._price_and_rank(ranked) if ranked else []

    def _capture(self, regions: list[Region]) -> None:
        """k input sets per workload, the boundary arrays of every viable
        region saved from a recorded pass at each."""
        self.session.wait_ready()
        for w in self.manifest.workloads:
            wanted: set[int] = set()
            for r in regions:
                for _label, m, _copies in capture_instances(r, self.traces):
                    if m.workload == w.name:
                        wanted |= set(m.input_ids) | set(m.output_ids)
            if not wanted:
                continue
            seeds = workload_seeds(self.manifest.seed, w.name, manifest_mod.BOUNDARY_INPUT_SETS)
            for si, seed in enumerate(seeds):
                tensors = materialize(w, self.manifest.primary, seed)
                arrays = self.session.off_clock(lambda: capture_boundaries(
                    self.tracer, self.baseline_model, tensors, self.traces[w.name], wanted))
                for r in regions:
                    for label, m, _copies in capture_instances(r, self.traces):
                        if m.workload != w.name:
                            continue
                        try:
                            ins = {a: arrays[a] for a in m.input_ids}
                            outs = {a: arrays[a] for a in m.output_ids}
                        except KeyError as e:
                            # a capture gap costs this region, never the job
                            r.rejected = f"capture missed boundary array {e}"
                            break
                        self.store.save(r.fingerprint, label, si, "inputs", ins)
                        self.store.save(r.fingerprint, label, si, "outputs", outs)
            self._capture_sweep(w, [r for r in regions if not r.rejected])
        self._validate_boundaries(regions)

    def _validate_boundaries(self, regions: list[Region]) -> None:
        """Verify each wave's contracts and saved sets before pricing or search."""
        from .regions.fingerprint import boundary_roles
        from .regions.store import BoundaryMismatch
        for region in regions:
            if region.rejected:
                continue
            rep = region.members[0]
            expected = boundary_roles(self.traces[rep.workload], rep)
            for member in region.members:
                if boundary_roles(self.traces[member.workload], member) != expected:
                    raise BoundaryMismatch(f"Region {region.fingerprint} groups incompatible "
                                           f"input/output roles at {member.workload}:{member.start_seq}")
            instances = [(label, member, manifest_mod.BOUNDARY_INPUT_SETS)
                         for label, member, _ in capture_instances(region, self.traces)]
            for label, span, _ in self._sweep_instances(region):
                if boundary_roles(self.sweep_traces[label], span) != expected:
                    raise BoundaryMismatch(f"Region {region.fingerprint} has incompatible "
                                           f"input/output roles at sweep {label}")
                instances.append((label, span, 1))
            for label, member, count in instances:
                for si in range(count):
                    self.store.validate(region.fingerprint, label, si, "inputs", member.input_ids)
                    self.store.validate(region.fingerprint, label, si, "outputs", member.output_ids)

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
            arrays = self.session.off_clock(lambda: capture_boundaries(
                self.tracer, self.baseline_model, self.sweep_tensors[label], retrace, wanted))
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
        if self.use_library_inference:
            return lambda: model(*tensors)
        if (baseline or self.baseline) == "compiled":
            compiled = mx.compile(lambda *t: model(*t))
            self.session.timed(lambda: compiled(*tensors))  # compile now; account for cooling debt
            return lambda: compiled(*tensors)
        return lambda: model(*tensors)

    def _baseline_arm(self):
        """What every whole-model number is measured against: the untouched
        model, or under a compiled baseline the same model with its baseline
        scopes compiled and empty. Correctness always uses the untouched one."""
        return self.baseline_model if self._compiled_baseline is None else self._compiled_baseline

    def _kernels_installed(self) -> bool:
        """Whether any installed wrapper carries a cut; a compiled baseline's
        empty wrappers alone are not an improvement to check or ship."""
        return any(not isinstance(state, tuple) or any(state[1].values()) for state in self.installed.values())

    def _timed_arms(self, incumbent=None):
        """Fresh timing closures for every declared workload."""
        incumbent = self._baseline_arm() if incumbent is None else incumbent
        return {w.name: (self._step_fn(incumbent, self.tensors[w.name]),
                         self._step_fn(self.model, self.tensors[w.name]))
                for w in self.manifest.workloads}

    def _copy_incumbent(self, emitted=None, overrides=None):
        """A fresh model carrying the given installed wrappers, with overrides
        (scope -> (wrapper, kernel specs)) in place of whatever those scopes
        and their descendants would have carried."""
        emitted = self.emitted if emitted is None else emitted
        overrides = overrides or {}
        if not emitted and not overrides:
            return self.baseline_model
        model = self.build_model()
        require_independent_models(self.model, model)
        require_independent_models(self.baseline_model, model)
        share_weights(self.baseline_model, model)
        for path, wrapper in emitted.items():
            if any(path == root or path.startswith(root + ".") for root in overrides):
                continue
            swap_install(model, path, _load_class(wrapper)(_resolve(model, path), self.installed[path][2]))
        for path, (wrapper, specs) in overrides.items():
            swap_install(model, path, _load_class(wrapper)(_resolve(model, path), specs))
        mx.clear_cache()
        return model

    def _identity(self, path: str, name: str):
        """The scope compiled with no cut, over every recorded call of it;
        NotReplayable names why the scope cannot be compiled."""
        from .bind.graph import emit_graph_wrapper_variants
        definitions = [(trace, sc, []) for trace in self.traces.values()
                       for sc in trace.scope_calls if sc.address.rsplit("@", 1)[0] == path]
        return emit_graph_wrapper_variants(definitions, name)

    def _settle_delivery(self, regions: list[Region]) -> None:
        """Before any clock: certify the identity of every scope planned for
        graph delivery, so the plan every clock follows is the delivery that
        will ship (a scope whose compiled identity is not bitwise the original
        moves its regions to replay), then clock the untouched model against
        itself with the outermost certified scopes compiled and empty. That
        is the compiled clock: a resolved slowdown sends those scopes to
        replay too; otherwise, when the manifest asks for the compiled
        baseline under library inference, that model becomes the baseline,
        installed empty so every kernel composes into it."""
        if self.baseline == "compiled":
            return
        planned: dict[str, list] = {}
        for region in regions:
            for member in region.members:
                if region.rejected or region.delivery.get(member.workload) != "graph":
                    continue
                scope = find_scope_call(self.traces[member.workload], member.scope_stack)
                planned.setdefault(scope.address.rsplit("@", 1)[0], []).append((region, member.workload))
        identity = {}
        for path in sorted(planned):
            try:
                identity[path] = self._identity(path, f"Id_{_safe(path)}")
            except NotReplayable as error:
                self._fall_back(planned[path], path, str(error))
        from autotuner_runtime.state import correctness_call
        runs = [lambda t=self.tensors[w.name]: correctness_call(self.baseline_model, t)
                for w in self.manifest.workloads]
        while identity:
            wrappers = {path: _load_class(e)(_resolve(self.baseline_model, path), {})
                        for path, e in identity.items()}
            check = self.session.off_clock(lambda: certify_identities(
                model=self.baseline_model, wrappers=wrappers, runs=runs), defer_cooling=True)
            if check.ok:
                break
            failed = check.scope if check.scope in identity else next(iter(identity))
            identity.pop(failed)
            self._fall_back(planned[failed], failed, check.reason)
        # Two compiled scopes cannot nest over one captured state, so the
        # outermost certified scopes stand in for all beneath them.
        outermost = {p: (e, {}) for p, e in identity.items()
                     if not any(q != p and p.startswith(q + ".") for q in identity)}
        slower = False
        if outermost:
            self.session.wait_ready()
            compiled = self._copy_incumbent({}, outermost)
            for w in self.manifest.workloads:
                comparison = compare(self.session, self._step_fn(self.baseline_model, self.tensors[w.name]),
                                     self._step_fn(compiled, self.tensors[w.name]),
                                     pairs=SHIP_PAIRS, defer_cooling=True)
                slower |= comparison.loses_by(0.0)
                clocks = self.report.baseline.setdefault("clocks_ms", {}).setdefault(
                    w.name, {"plain": self.step_ms.get(w.name)})
                clocks["compiled"] = statistics.median(comparison.candidate_ms)
                clocks["compiled_vs_plain"] = {"ratio": comparison.median_ratio, "stability": comparison.stability,
                                               "verdict": ("resolved_improvement" if comparison.wins_by(0.0) else
                                                           "resolved_regression" if comparison.loses_by(0.0)
                                                           else "inconclusive")}
                fallbacks = [p for p in outermost if (wrapper := _resolve(compiled, p))
                             and (wrapper._graph_fallbacks or wrapper._graph_fallback_reason)]
                if fallbacks:
                    raise RuntimeError(f"the compiled clock ran the original module at {fallbacks}")
        if slower:
            # Compiling these scopes costs the step more than fusion saves
            # (each compiled call has a host price): replay delivers here.
            for path in list(identity):
                self._fall_back(planned[path], path, "compiling the scope makes the step slower")
            identity.clear()
            outermost.clear()
        self.certified_scopes.update(identity)
        adopt = bool(outermost) and self.use_library_inference and self.manifest.baseline == "compiled"
        self.report.baseline.update(compiled_available=bool(outermost), choice="compiled" if adopt else "plain",
                                    reason=None if adopt or not outermost else self.report.baseline.get("reason"))
        if adopt:
            # The baseline every win is measured against is this model:
            # compiled where it can be, empty. Installed as the starting
            # state, every kernel composes into it, and the artifact ships it.
            self.baseline = "compiled"
            self._compiled_baseline = compiled
            for w in self.manifest.workloads:
                self.step_ms[w.name] = self.report.baseline["clocks_ms"][w.name]["compiled"]
                self.report.step_ms[w.name] = {"before": self.step_ms[w.name]}
            for path, (emitted, _specs) in outermost.items():
                original = _resolve(self.model, path)
                swap_install(self.model, path, _load_class(emitted)(original, {}))
                variants = {(name, sc.address): [] for name, trace in self.traces.items()
                            for sc in trace.scope_calls if sc.address.rsplit("@", 1)[0] == path}
                self.installed[path] = (original, variants, {})
                self.emitted[path] = emitted
        self.log.append("delivery_settled", graph=sorted(identity), replay=sorted(self.replay_scopes),
                        baseline=self.baseline, baseline_scopes=sorted(outermost) if adopt else [])

    def _fall_back(self, regions, path: str, reason: str) -> None:
        """Graph delivery declined for a scope: its regions take replay there."""
        self.replay_scopes.add(path)
        self.log.append("graph_fallback", scope=path, reason=reason)
        for region, workload in regions:
            region.delivery[workload] = "replay"
            region.delivery_reasons[workload] = reason

    def _clock_steps(self) -> None:
        """The step clocks on every workload, and the baseline every share and
        win is measured against. A step that keeps Python state (a KV cache)
        cannot be compiled from outside the model: mx.compile swaps only state
        handed to it in a dict or list, and one compiled call would leave the
        model holding tracers. Such a step gets the plain baseline, and its
        compiled clock is never taken. Where the compiled clock is taken, the
        model is checked alive and unchanged afterwards, because a compiled
        call on a state-keeping step returns a plausible number first and
        breaks the model only on the next call."""
        if self.use_library_inference:
            # The library runs the plain model. The compiled baseline, when
            # requested, is the same model with its outermost compilable
            # scopes compiled and empty, settled with delivery below.
            self.baseline = "plain"
            self.report.baseline = {
                "requested": self.manifest.baseline, "choice": "plain",
                "execution": "library_generation", "compiled_available": False,
                "reason": "the compiled baseline is settled once delivery is",
                "clocks_ms": {},
            }
            self.log.append("baseline", **self.report.baseline)
            for w in self.manifest.workloads:
                clock = step_clock(self.session, self._step_fn(self.baseline_model, self.tensors[w.name]))
                self.step_ms[w.name] = clock.median_ms
                self.report.step_ms[w.name] = {"before": clock.median_ms}
                self.report.baseline["clocks_ms"][w.name] = {"plain": clock.median_ms, "compiled": None}
                self.log.append("step_clock", workload=w.name, phase="before", baseline="library_generation",
                                median_ms=clock.median_ms, generated_tokens=self.manifest.final_benchmark.steps)
            return
        kept = {w: marks for w, t in self.traces.items() if (marks := _state_marks(t))}
        requested = self.manifest.baseline
        self.baseline = "plain" if kept else requested
        reason = None if not kept else (
            "the step keeps Python state (" +
            "; ".join(f"{w}: {marks}" for w, marks in kept.items()) +
            "); a compiled call would leave the model holding tracers")
        self.report.baseline = {"requested": requested, "choice": self.baseline,
                                "compiled_available": not kept, "reason": reason, "clocks_ms": {}}
        self.log.append("baseline", requested=requested, choice=self.baseline,
                        compiled_available=not kept, reason=reason)
        for w in self.manifest.workloads:
            print(f"baseline: timing {w.name} ({self.baseline})", flush=True)
            tensors = self.tensors[w.name]
            clocks = {"plain": step_clock(self.session,
                                          self._step_fn(self.model, tensors, "plain")).median_ms,
                      "compiled": None}
            if self.baseline == "compiled":
                # mx.array copies: the step may return a buffer it writes
                before = [mx.array(a) for a in flatten_arrays(self.model(*tensors))]
                mx.eval(before)
                clocks["compiled"] = step_clock(
                    self.session, self._step_fn(self.model, tensors, "compiled")).median_ms
                self._assert_survived_compile(w.name, tensors, before)
            self.report.baseline["clocks_ms"][w.name] = clocks
            self.step_ms[w.name] = clocks[self.baseline]
            self.report.step_ms[w.name] = {"before": clocks[self.baseline]}
            self.log.append("step_clock", workload=w.name, phase="before", baseline=self.baseline,
                            median_ms=clocks[self.baseline], plain_ms=clocks["plain"],
                            compiled_ms=clocks["compiled"])
    def _record_step_floor(self) -> None:
        """The scout line: how much of each step is physics no kernel can
        touch, and how much is room. Needs both the step clock and the chip's
        peaks, so it runs after both and constrains the order of neither."""
        if self.peaks is None:  # a job measures the chip; a test may not
            return
        for w in self.manifest.workloads:
            trace = self.traces[w.name]
            floor = step_floor(trace, self.peaks, self.step_ms[w.name])
            self._raise_peaks((n for n in trace.nodes if n.seq not in trace.dead),
                              self.step_ms[w.name], f"the {w.name} step", floor["bytes_mb"] * 1e6)
            floor = step_floor(trace, self.peaks, self.step_ms[w.name])
            self.report.coverage.setdefault("step_floor", {})[w.name] = floor
            self.log.append("step_floor", workload=w.name, **floor)

    def _assert_survived_compile(self, workload: str, tensors, before) -> None:
        """A compiled call on a step that keeps state the detector missed does
        not raise: it returns a plausible number and leaves the model holding
        tracers, so every later win would be measured against garbage. Take
        the same step again and demand the same bits."""
        advice = ("the step keeps state the trace did not show; rerun with baseline: plain "
                  "in the manifest and report the model to the maintainer")
        try:
            after = flatten_arrays(self.model(*tensors))
            mx.eval(after)
        except Exception as e:
            raise RuntimeError(f"the compiled baseline broke the model on {workload}: {e}; {advice}") from e
        if len(after) != len(before) or not all(bitwise_equal(a, b) for a, b in zip(after, before)):
            raise RuntimeError(f"the compiled baseline changed what {workload} returns; {advice}")

    def _price_and_rank(self, regions: list[Region]) -> list[Region]:
        """Each region's share of the step, its physical limit, the share
        floor, and the ranking."""
        self.session.wait_ready()
        step_fns = {w.name: self._step_fn(self.model, self.tensors[w.name])
                    for w in self.manifest.workloads}
        survivors = [r for r in regions if not r.rejected]
        self.log.append("pricing", regions=len(survivors), wave=self.selection_wave)
        print(f"pricing: {len(survivors)} regions sharing paired model measurements", flush=True)
        price_group(survivors, self.session, self.traces, self.store, step_fns,
                    baseline=self.baseline, pairs=PRICE_PAIRS)
        for r in survivors:
            self.log.append("priced", fingerprint=r.fingerprint, p=r.p, region_ms=r.t_rep_ms,
                            price_stability=dict(r.stability))
        for r in survivors:
            if r.rejected:
                continue
            rep = r.members[0]
            # one copy's cost against one copy's floor, both from one paired
            # window, so nothing the machine did between pricing and the peak
            # readings can open headroom that is not there
            by_workload = {}
            for label, member, copies in capture_instances(r, self.traces):
                price = r.prices[label]
                trace = self.traces[member.workload]
                if self.peaks is not None:  # a job measures the chip; a test may not
                    self._raise_peaks(trace.nodes[member.start_seq:member.end_seq + 1], price.ms,
                                      f"region {r.fingerprint[:8]} at {label}")
                roof = stretch_roofline(trace, member, self.peaks, t_orig_ms=price.ms,
                                        floor_ms=price.floor_ms, rates=price.gflops)
                by_workload.setdefault(member.workload, []).append((roof, copies))
                if member is rep:
                    r.roofline = roof
            for workload, roofs in by_workload.items():
                total_floor = sum(roof.t_roofline_ms * copies for roof, copies in roofs)
                r.rooflines[workload] = replace(
                    roofs[0][0], s_max=r.t_orig_ms[workload] / max(total_floor, 1e-9))
        kept = rank(apply_floor(survivors))
        latest = {row["fingerprint"]: row for row in self.report.pricing}
        for r in survivors:
            latest[r.fingerprint] = {
                "fingerprint": r.fingerprint, "ops": list(r.ops), "copies": r.copies,
                "p": dict(r.p), "region_ms": dict(r.t_rep_ms),
                "ranking_basis": "estimated_removable_share",
                "price_stability": dict(r.stability),
                "estimate_basis": "boundary_probe_and_known_compute",
                "compute_model_complete": "metal_kernel" not in r.ops,
                "headroom_is_estimate": True,
                "estimated_s_max": {w: roof.s_max for w, roof in r.rooflines.items()},
                "measured_compute_gflops": {label: price.gflops for label, price in r.prices.items()},
                "potential_model_latency_reduction_pct": {
                    w: 100 * share for w, share in r.removable_p.items()},
                "status": "rejected" if r.rejected else "eligible",
                "reason": r.rejected,
            }
        self.report.pricing = sorted(latest.values(),
                                     key=lambda row: -sum(row["p"].values()))
        # every compute op belongs to exactly one atomic region, so their
        # shares sum to the fraction of the step that any candidate can reach;
        # the rest runs inside stranded scopes or ops no kernel can replace
        self.report.coverage.update({
            "measured_atomic_share_this_wave": {
                w.name: sum(r.p.get(w.name, 0.0) for r in regions if self._atomic(r))
                for w in self.manifest.workloads
            },
            "step_ms": dict(self.step_ms),
        })
        for r in regions:
            if r.rejected:  # share-floor cuts belong in the report
                self.report.stranded.append({"fingerprint": r.fingerprint, "ops": list(r.ops),
                                             "reason": r.rejected})
        self.log.append("ranked", kept=len(kept))
        self.report.write(self.work_dir / "report.json")
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
        """Record chip throughput and require the timing control to pass.

        Utilization and peak estimates are informational. A significant
        difference between identical arms means the clock cannot currently
        distinguish optimization from bias, so it gets one retry, then stops.
        """
        print("calibration: measuring bandwidth, compute throughput and timing noise", flush=True)
        busy = self.gpu_busy_at_start
        self.report.session["gpu_utilization_at_start_pct"] = busy
        if busy is not None and busy > BUSY_GPU_PERCENT:
            self._env_warning(f"macOS reports {busy:.0f}% GPU utilization before the job; "
                              "this counter does not measure available throughput. "
                              "Measured throughput and paired timings determine the results")
        self._record_peaks(measure_peaks(self.session), "job_start")
        reason = peaks_implausible(self.peaks)
        if reason:
            self._env_warning(f"{reason}; the chip is throttled or shared, so small wins may go "
                              "unnoticed and the room line will read high")
        for attempt in (1, 2):
            floor = aa_null(self.session, pairs=8)
            passed = not (floor.wins_by(0.0) or floor.loses_by(0.0))
            self.report.session["aa_floor_sigma_ms"] = floor.sigma_ms
            self.report.session["aa_control"] = {"passed": passed, "attempts": attempt}
            self.log.append("aa_floor", attempt=attempt, passed=passed,
                            sigma_ms=floor.sigma_ms, median_delta_ms=floor.median_delta_ms,
                            stability=round(floor.stability, 3))
            if passed:
                break
            if attempt == 1:
                self._env_warning(f"the A/A control found a {floor.median_delta_ms:+.4f} ms "
                                  "difference between identical code; this can create false wins. "
                                  "Retrying the timing control once")
            else:
                raise RuntimeError("A/A timing control failed twice: identical code showed a "
                                   "significant timing difference. Measurements are not trustworthy; "
                                   "the run stopped before searching kernels")

    def _raise_peaks(self, nodes, measured_ms: float, source: str, bytes_moved: float = 0.0) -> None:
        """The chip seen doing these ops, or moving these bytes, faster than
        the peaks allow raises them: a ceiling never sits under a measurement."""
        raised = observed_peaks(self.peaks, nodes, measured_ms, bytes_moved)
        if raised is not self.peaks:
            self._record_peaks(raised, f"raised by {source}")

    def _record_peaks(self, peaks, when: str) -> None:
        self.peaks = peaks
        self.report.peaks = {"bandwidth_gbps": peaks.bandwidth_gbps,
                             "flops_gflops": peaks.flops_gflops, "launch_us": peaks.launch_us}
        self.log.append("peaks", when=when, **self.report.peaks)

    def _env_warning(self, detail: str) -> None:
        self.log.append("env_warning", detail=detail)
        print(f"WARNING: {detail}")

    # -- stage 3: one region --------------------------------------------------

    def _contract(self, region: Region) -> RegionContract:
        rep = region.members[0]
        specs = self.traces[rep.workload].span_specs(rep.start_seq, rep.end_seq)
        scaffold = self._build_scaffold(region)
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
            native_call=scaffold.native_call,
            input_signature=scaffold.input_signature,
        )

    def _eval_sets(self, region: Region) -> list[EvalSet]:
        sets = []
        for label, m, _copies in capture_instances(region, self.traces):
            k = self.store.set_count(region.fingerprint, label)
            if region.t_orig_ms.get(m.workload) is None:
                # open_region screens this; backstop so it can never reach
                # ladder validation as a bare None
                raise RuntimeError(f"region eval set unpriced for workload {m.workload!r}")
            sets.append(EvalSet(
                label=label,
                inputs_paths=[str(self.store._path(region.fingerprint, label, i, "inputs"))
                              for i in range(k)],
                reference_paths=[str(self.store._path(region.fingerprint, label, i, "outputs"))
                                 for i in range(k)],
                t_library_ms=region.prices[label].ms if label in region.prices
                    else region.t_rep_ms.get(m.workload),
                correctness_only=False,
                nodes_json=nodes_to_json(self.traces[m.workload].nodes[m.start_seq:m.end_seq + 1]),
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

    def _target_workload(self, region: Region, requested: str | None = None) -> str:
        """Choose before timing, independently of manifest order."""
        if requested is not None:
            if requested not in region.workloads:
                raise ValueError(f"target_workload {requested!r} must be one of {sorted(region.workloads)}")
            return requested
        return min(region.workloads, key=lambda w: (
            -region.removable_p.get(w, 0.0), -region.p.get(w, 0.0), w))

    def _ladder_job(self, region: Region, kernel: KernelSpec, assoc_tag: str,
                    run_clock: bool, target_workload: str | None = None) -> LadderJob:
        target = self._target_workload(region, target_workload)
        label, rep, copies = next(instance for instance in capture_instances(region, self.traces)
                                   if instance[1].workload == target)
        trace = self.traces[rep.workload]
        nodes = trace.nodes[rep.start_seq:rep.end_seq + 1]
        contract = self._contract(region)
        eval_sets = self._eval_sets(region)
        eval_sets.sort(key=lambda es: es.label != label)
        # Correctness still visits all cases, even when the timed one is small.
        slowest_copy_ms = max((es.t_library_ms or 0.0 for es in eval_sets), default=0.0)
        roof = region.rooflines.get(target, region.roofline)
        return LadderJob(
            baseline=region.library_arm(target, self.baseline),
            kernel=kernel,
            contract=contract,
            assoc_tag=assoc_tag,
            nodes_json=nodes_to_json(nodes),
            input_ids=rep.input_ids,
            output_ids=rep.output_ids,
            eval_sets=eval_sets,
            tolerances=self.manifest.tolerances,
            min_win_ms=MIN_WIN_MS / max(copies, 1),
            run_clock=run_clock,
            clock_pairs=self.clock_pairs,
            # the timing child runs a few hundred passes under pacing; a region
            # whose one pass takes hundreds of ms needs minutes, not a fixed cap
            timeout_s=120.0 + 0.8 * slowest_copy_ms,
            weight_inputs=tuple(a in trace.weights for a in rep.input_ids),
            compute_floor_ms=roof.t_compute_ms if roof else 0.0,
        )

    def _kernel_from_proposal(self, run: RegionRun, region: Region, proposal, hyp_id: str) -> KernelSpec:
        kid = _kernel_id(region, hyp_id)
        parent = resolve_parent(run, proposal.parent_kernel_id)
        return kernel_from_proposal(self._contract(region), parent, proposal, kid)

    def _evaluate_kernel(self, region, kernel, assoc_tag, run_clock=True, target_workload=None):
        try:
            target = self._target_workload(region, target_workload)
        except ValueError as error:
            return LadderResult("failed", "static", {"reason": str(error)}, None, None, None, None, [])
        job = self._ladder_job(region, kernel, assoc_tag, run_clock, target)
        job.timing_incumbent = next((cut.kernel
            for _original, variants, _kernels in self.installed.values()
            for cuts in variants.values() for cut in cuts
            if cut.fingerprint == region.fingerprint), None)
        result = run_ladder(job, session=self.session)
        result.detail.update(target_workload=target, timing_case=job.eval_sets[0].label)
        # Preserve completed gates before whole-model checks can be interrupted.
        self.log.append("ladder_result", fingerprint=region.fingerprint,
                        kernel=kernel.kernel_id, assoc_tag=assoc_tag, result=asdict(result))
        return result

    def open_region(self, region: Region, judge) -> RegionRun:
        from .scaffold import NoScaffold

        run = RegionRun(region=region)
        run.directions = directions_for(region.roofline.bound if region.roofline else None,
                                        self.manifest.seed, region.fingerprint)
        for m in region.members:
            if region.t_orig_ms.get(m.workload) is None:
                run.close_rule = f"workload {m.workload!r} was never priced for this region"
                self.log.append("region_skip", fingerprint=region.fingerprint,
                                reason=run.close_rule)
                return run
        rep = region.members[0]
        trace = self.traces[rep.workload]
        try:
            scaffold = self._build_scaffold(region)
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
        result = self._evaluate_kernel(region, scaffold, "preserving", run_clock=scaffold.reference_sequence is None)
        if (result.outcome == "failed" and scaffold.reference_sequence is None
                and result.failed_gate not in ("static", "compile")):
            # A generated starter can change rounding before search even begins.
            # Keep the original sequence as an exact seed so the judge can still
            # propose an explicitly changing replacement for this region.
            from .scaffold.native import reference_sequence_seed
            try:
                original = self._rename(reference_sequence_seed(trace, rep), region, "scaffold")
            except NoScaffold:
                original = None
            if original is not None:
                self.scaffold_overrides[region.fingerprint] = original
                original_result = self._evaluate_kernel(region, original, "preserving", run_clock=False)
                if original_result.outcome != "failed":
                    self.log.append("scaffold_reference_fallback", fingerprint=region.fingerprint,
                                    failed_gate=result.failed_gate, detail=result.detail)
                    scaffold, result = original, original_result
                    run.kernels[scaffold.kernel_id] = scaffold
                    write_kernel(self.kernel_dir, scaffold)
                else:
                    self.scaffold_overrides.pop(region.fingerprint, None)
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
        # the starting kernel is judged on the whole model like any edit; a
        # naive scaffold rarely wins, but a stitched one at library parity can
        promotion = self._bind_and_promote(run, scaffold, result)
        _promotion_feedback(result, promotion)
        if promotion:
            run.shipped, run.shipped_ms, run.shipped_ratio = scaffold, result.region_ms, _ratio(result)
            run.shipped_floor_ms = result.floor_ms
            run.shipped_workload = result.detail.get("target_workload")
            outcome = "shipped"
        else:
            outcome = promotion.outcome
            if outcome == "rolled_back":
                # Keep the rejected source for repair, but make the default
                # parent an original implementation with a matching contract.
                from .scaffold.native import native_seed, reference_sequence_seed
                prior_override = self.scaffold_overrides.get(region.fingerprint)
                try:
                    node = trace.nodes[rep.start_seq]
                    build_original = (native_seed if rep.start_seq == rep.end_seq
                                      and node.kernel_definition is not None else reference_sequence_seed)
                    original = self._rename(build_original(trace, rep), region, "original")
                except (NoScaffold, ValueError, KeyError, AttributeError) as error:
                    original = None
                    run.close_rule = f"whole-model scaffold rejection; original starter unavailable: {error}"
                if original is not None:
                    self.scaffold_overrides[region.fingerprint] = original
                    original_result = self._evaluate_kernel(region, original, "preserving", run_clock=False)
                    if original_result.outcome == "failed":
                        run.close_rule = ("whole-model scaffold rejection; original starter failed "
                                          f"{original_result.failed_gate}: {_short_reason(original_result.detail)}")
                    else:
                        run.scaffold = original
                        run.kernels[original.kernel_id] = original
                        write_kernel(self.kernel_dir, original)
                        self._set_head(run, original, original_result, "preserving")
                        self.log.append("scaffold_model_fallback", fingerprint=region.fingerprint,
                                        rejected_kernel=scaffold.kernel_id, original_kernel=original.kernel_id)
                if run.close_rule is not None:
                    if prior_override is None:
                        self.scaffold_overrides.pop(region.fingerprint, None)
                    else:
                        self.scaffold_overrides[region.fingerprint] = prior_override
                    run.scaffold = run.head = None
                    run.head_ms = run.head_ratio = run.head_floor_ms = None
                    self.log.append("region_skip", fingerprint=region.fingerprint, reason=run.close_rule)
        repaired = scaffold is not run.kernels.get(_kernel_id(region, "scaffold"))
        self._record_attempt(
            run, "scafix" if repaired else "scaffold", "fix" if repaired else "scaffold",
            "the judge's one fix of the starting kernel" if repaired else "the harness's starting kernel",
            "preserving", scaffold, None, result, outcome=outcome)
        return run

    def _build_scaffold(self, region: Region):
        if region.fingerprint in getattr(self, "scaffold_overrides", {}):
            return self.scaffold_overrides[region.fingerprint]
        from .scaffold import NoScaffold, build_scaffold
        from .ladder.static_checks import METAL_BUFFER_LIMIT, buffer_count
        instances = [(self.traces[m.workload], m)
                     for _label, m, _copies in capture_instances(region, self.traces)]
        for workload in self.manifest.workloads:
            rep = next((m for m in region.members if m.workload == workload.name), None)
            if rep is None:
                continue
            for label, _dims in self._sweep_points(workload):
                trace = self.sweep_traces[label]
                instances.append((trace, locate_span(self.traces[workload.name], rep,
                                                     trace, workload.name)))
        shapes = [[trace.span_specs(m.start_seq, m.end_seq)[a][0] for a in m.input_ids]
                  for trace, m in instances]
        rep = region.members[0]
        scaffold = build_scaffold(self.traces[rep.workload], rep, shapes)
        buffers = buffer_count(scaffold, [len(shape) for shape in shapes[0]])
        if buffers > METAL_BUFFER_LIMIT:
            raise NoScaffold("buffer-limit", f"starter needs {buffers} Metal buffer arguments; "
                             f"the limit is {METAL_BUFFER_LIMIT}")
        return scaffold

    def _rename(self, spec: KernelSpec, region: Region, tag: str) -> KernelSpec:
        kid = _kernel_id(region, tag)
        d = {k: getattr(spec, k) for k in spec.__dataclass_fields__}
        d["kernel_id"] = kid
        d["name"] = f"at_{kid}"
        return KernelSpec(**d)

    def _judge_fix(self, run: RegionRun, judge, scaffold, result):
        """The spec's one repair attempt on a starting kernel that failed its
        own checks. Returns (kernel, ladder result) or None."""
        if self._close_rule(run) is not None:
            return None
        self._spend(run)
        region = run.region
        run.head, run.last_kernel = scaffold, scaffold.kernel_id
        verdict = _verdict_payload("scaffold", scaffold.kernel_id, "failed", result)
        writing_for = {"id": "scafix", "kind": "fix", "assoc_tag": "preserving",
                       "hypothesis": "repair the starting kernel so it passes the checks"}
        meta = self._meta(run, Queue(), writing_for)
        resp, error = self._ask_judge(run, "scaffold_fix", lambda: judge.next(meta, verdict))
        if error is not None:
            self.log.append("scaffold_fix", fingerprint=region.fingerprint,
                            outcome="judge_error", reason=error)
            return None
        self._note_lesson(run, resp)
        if resp.kernel is None:
            self.log.append("scaffold_fix", fingerprint=region.fingerprint, outcome="yield")
            return None
        fixed = self._kernel_from_proposal(run, region, resp.kernel, "scafix")
        run.kernels[fixed.kernel_id] = fixed
        write_kernel(self.kernel_dir, fixed)
        target = {"target_workload": resp.kernel.target_workload} if resp.kernel.target_workload else {}
        check = self._evaluate_kernel(region, fixed, "preserving", **target)
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
        decides; repeat. A reply with nothing to evaluate is asked again once
        with the reason, then each further one costs an attempt, so the
        budget is spent by the judge and never handed back."""
        rule = self._close_rule(run)
        if rule:
            run.close_rule = rule
            return
        region = run.region
        if run.queue is None and getattr(judge, "combined_start", False):
            run.queue = Queue()  # the normal mutation/proposal protocol also starts a search
        if run.queue is None:
            run.queue = Queue()
            queue = run.queue
            verdict = run.last_verdict
            meta = self._meta(run, queue, None)
            resp, failure = self._ask_judge(run, "seed", lambda: judge.seed(meta))
            problem = None
            if resp is not None:
                self._note_lesson(run, resp)
                try:
                    queue.seed(resp.queue)
                except QueueError as e:
                    problem = f"your seed queue was refused: {e}"
            if resp is None or problem:
                verdict, close = self._empty_reply(run, None, failure, problem, verdict)
                run.last_verdict = verdict
                if close:
                    run.close_rule = close
                    return
        queue, verdict = run.queue, run.last_verdict

        while True:
            rule = self._close_rule(run)
            if rule:
                run.close_rule = rule
                return
            front = queue.peek_ready()
            meta = self._meta(run, queue, _item_view(front) if front else None)
            resp, failure = self._ask_judge(run, "next", lambda: judge.next(meta, verdict))
            item = problem = None
            if resp is not None:
                run.errors = 0
                self._note_lesson(run, resp)
                try:
                    queue.apply_mutations(resp.mutations)
                except QueueError as e:
                    problem = f"your plan edits were refused and none applied: {e}"
                else:
                    if resp.kernel is None:
                        problem = "a yield is refused while the budget lasts"
                    else:
                        item = queue.ready_item(resp.kernel.item_id)
                        if item is None:
                            named = resp.kernel.item_id
                            problem = (f"your kernel is for {named!r}, which is not a ready queued item"
                                       if named else "no queued item is ready for your kernel") + \
                                      f" (queued: {', '.join(queue.ids()) or 'nothing'})"
                        else:
                            problem = self._opener_rule(run, item, resp.kernel.parent_kernel_id)
                            if problem is None:
                                queue.pop_ready(item.id)
                            else:
                                item = None
            if item is None:
                verdict, close = self._empty_reply(run, front, failure, problem, verdict)
                if close:
                    run.close_rule = close
                    return
                continue
            run.refused = 0
            self._spend(run)
            parent_spec = resolve_parent(run, resp.kernel.parent_kernel_id)
            parent = parent_spec.kernel_id if parent_spec else resp.kernel.parent_kernel_id
            if parent_spec is run.scaffold and len(run.openers) < self._openers(run):
                run.openers.append(item.kind)
            kernel = self._kernel_from_proposal(run, region, resp.kernel, item.id)
            run.kernels[kernel.kernel_id] = kernel
            write_kernel(self.kernel_dir, kernel)
            if parent_spec is not None:
                target = {"target_workload": resp.kernel.target_workload} if resp.kernel.target_workload else {}
                result = self._evaluate_kernel(region, kernel, item.assoc_tag, **target)
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
        later call carries the newest ones."""
        if resp.lesson:
            excerpt = resp.lesson
            if len(excerpt) > LESSON_CONTEXT_CHARS:
                excerpt = excerpt[:LESSON_CONTEXT_CHARS - 3].rstrip() + "..."
            self.lessons.append({"region": run.region.fingerprint[:8], "ops": list(run.region.ops),
                                 "lesson": excerpt})
            del self.lessons[:-LESSONS_KEPT]
            self.log.append("lesson", fingerprint=run.region.fingerprint, lesson=resp.lesson)

    def _budget(self, run: RegionRun) -> dict:
        """Attempts left for the region and for the job: what the judge is
        told and what the close rule reads."""
        return {"attempts_left_region": self.manifest.budget_per_region - run.hypotheses,
                "attempts_left_job": self.manifest.budget_total - self.total_hypotheses}

    def _spend(self, run: RegionRun) -> None:
        run.hypotheses += 1
        self.total_hypotheses += 1

    def _openers(self, run: RegionRun) -> int:
        """How many openers the region's widening round holds."""
        return min(OPENERS, len(run.directions))

    def _opener_rule(self, run: RegionRun, item, parent_name: str) -> str | None:
        """While the widening round lasts, a kernel is an opener, written
        against the scaffold under a kind no earlier opener used, or a repair
        of a kernel that failed. Returns the refusal, or None. A parent that
        names nothing is left to the verdict that says so."""
        left = self._openers(run) - len(run.openers)
        if left <= 0:
            return None
        parent = resolve_parent(run, parent_name)
        if parent is None:
            return None
        if parent is run.scaffold:
            if item.kind in run.openers:
                return (f"the widening round already opened {item.kind!r}; an opener needs a kind "
                        f"no earlier opener used (opened: {', '.join(run.openers)})")
            return None
        if run.attempts.get(parent.kernel_id, {}).get("verdict") in ("failed", "rolled_back"):
            return None
        return (f"the widening round has {left} opener(s) left, so {parent_name!r} is refused as a "
                f"parent: write against the scaffold under a kind no earlier opener used (see "
                f"directions), or repair a kernel that failed")

    def _empty_reply(self, run: RegionRun, front, failure: str | None, problem: str | None,
                     last: dict | None) -> tuple[dict, str | None]:
        """Bookkeeping for a reply that left nothing to evaluate: babble, a
        transport failure, or a refusal (a yield, refused plan edits, a
        kernel for an item that is not ready, a refused seed). Returns the
        verdict the next call carries; repeated transport failures stop the
        job. A babble costs an attempt, since the transport already asked
        once more; a refusal is asked again once for free, then charged."""
        if failure == "babble":
            self._spend(run)
            hyp = self._record_empty(run, front, "judge_babble", None)
            return {"hypothesis_id": hyp, "outcome": "failed", "failed_gate": "judge_babble"}, None
        if failure is not None:
            run.errors += 1
            hyp = self._record_empty(run, front, "judge_error", failure)
            verdict = {"hypothesis_id": hyp, "outcome": "failed", "failed_gate": "judge_error",
                       "detail": {"reason": failure}}
            if run.errors >= 3:
                raise RuntimeError(f"judge unavailable after 3 consecutive transport errors; "
                                   f"stopping the job instead of opening another region. Last error: {failure}")
            return verdict, None
        run.refused += 1
        if run.refused > 1:
            self._spend(run)
            self._record_empty(run, front, "plan_refused", problem)
        problem = f"{problem}; {self._budget(run)['attempts_left_region']} attempts remain in this region"
        self.log.append("plan_refused", fingerprint=run.region.fingerprint, reason=problem)
        return {**(last or {}), "plan_refused": problem}, None

    def _record_empty(self, run: RegionRun, front, gate: str, reason: str | None) -> str:
        """An attempt row for a reply with nothing to evaluate, under its own
        id: the queued item it was asked for stays queued and keeps its name."""
        run.empty_replies += 1
        hyp = f"{front.id if front else 'none'}_{gate}{run.empty_replies}"
        self._record_attempt(run, hyp, front.kind if front else "none",
                             front.hypothesis if front else "", None, None, None, None,
                             gate=gate, reason=reason)
        return hyp

    def _ask_judge(self, run: RegionRun, phase: str, call):
        """One judge call with its log row: (response, None), or (None, why)."""
        t0 = time.perf_counter()
        print(f"judge: {phase} for region {run.region.fingerprint[:8]}", flush=True)
        self.log.append("judge_pending", fingerprint=run.region.fingerprint, phase=phase)
        try:
            resp = call()
        except JudgeBabble:
            run.errors = 0  # a malformed answer still proves the transport answered
            self.log.append("judge", fingerprint=run.region.fingerprint, phase=phase,
                            latency_s=round(time.perf_counter() - t0, 1), action="babble")
            return None, "babble"
        except Exception as e:
            # Preserve the diagnostic; repeated transport failures stop the job
            # in _empty_reply without consuming optimization attempts.
            self.log.append("judge", fingerprint=run.region.fingerprint, phase=phase,
                            latency_s=round(time.perf_counter() - t0, 1),
                            action="error", reason=str(e))
            return None, f"{type(e).__name__}: {e}"
        summary = ({"queue": len(resp.queue)} if phase == "seed" else
                   {"mutations": len(resp.mutations),
                    "action": "yield" if resp.kernel is None else "proposal"})
        run.errors = 0
        self.log.append("judge", fingerprint=run.region.fingerprint, phase=phase,
                        latency_s=round(time.perf_counter() - t0, 1), **summary)
        return resp, None

    def _set_head(self, run: RegionRun, kernel: KernelSpec, clock, tag: str) -> None:
        """clock is a ladder result or a recorded attempt: anything with the
        kernel's region_ms and library_ms."""
        get = clock.get if isinstance(clock, dict) else lambda k: getattr(clock, k)
        run.head, run.head_tag = kernel, tag
        run.head_ms = get("region_ms")
        run.head_floor_ms = get("floor_ms")
        run.head_sigma_ms = get("sigma_ms")
        run.head_age = 0
        run.head_ratio = _ratio(clock)
        run.head_workload = (clock.get("target_workload") if isinstance(clock, dict)
                             else clock.detail.get("target_workload"))

    def _apply_verdict(self, run: RegionRun, item, kernel: KernelSpec, result) -> str:
        """Fail, climb, or ship. A correct kernel is installed and judged on
        the whole model, never on the region clock: the region clock cannot
        see the call site or the wrapper, so it stays informational and only
        picks head, the kernel the judge keeps editing. Its ratio to the
        library it was clocked beside compares across clocks taken minutes
        apart, which the running-time milliseconds do not."""
        run.head_age += 1
        at_floor = (None not in (result.region_ms, result.floor_ms, result.sigma_ms)
                    and result.region_ms - result.floor_ms <= result.sigma_ms)
        run.floor_streak = run.floor_streak + 1 if at_floor else 0
        if result.outcome == "failed":
            return "failed"
        promotion = self._bind_and_promote(run, kernel, result, assoc_tag=item.assoc_tag)
        _promotion_feedback(result, promotion)
        if promotion:
            run.shipped, run.shipped_ms, run.shipped_ratio = kernel, result.region_ms, _ratio(result)
            run.shipped_floor_ms = result.floor_ms
            run.shipped_workload = result.detail.get("target_workload")
            self._set_head(run, kernel, result, item.assoc_tag)
            return "shipped"
        if promotion.outcome == "rolled_back":
            return "rolled_back"  # retain the source for repair, keep the previous valid head
        # Correctness passed (or the region was not nominated for model testing).
        # Ratios from different workloads are not comparable. Keep this
        # attempt editable without pretending it beat a head timed elsewhere.
        if (run.head_ratio is None or run.head_workload != result.detail.get("target_workload")
                or _ratio(result) < run.head_ratio):
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
                "target_workload": detail.get("target_workload"),
                "timing_case": detail.get("timing_case"),
                "model_check": _safe_detail(detail.get("model_check", {})),
                "incumbent_screen": _safe_detail(detail.get("incumbent_screen", {})),
            }
        if detail.get("incumbent_screen", {}).get("resolved_regression"):
            how = "correct_slower: slower than the installed kernel in a fresh region comparison; model timing skipped"
        elif detail.get("model_check", {}).get("reason"):
            how = f"{outcome}: {detail['model_check']['reason']}"
        elif gate:
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
            target_workload=detail.get("target_workload"), timing_case=detail.get("timing_case"),
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
        for label, m, _copies in capture_instances(region, self.traces):
            specs = self.traces[m.workload].span_specs(m.start_seq, m.end_seq)
            io_specs[label] = {
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
                wanted.add(_kernel_id(region, item["depends_on"]))
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
            budget=self._budget(run),
            last_verdict=run.attempts.get(run.last_kernel) if run.last_kernel else None,
            writing_for=writing_for,
            default_target_workload=self._target_workload(region),
            head_workload=run.head_workload, shipped_workload=run.shipped_workload,
            directions=run.directions,
            widening={"openers": self._openers(run), "opened": list(run.openers),
                      "left": max(self._openers(run) - len(run.openers), 0)},
        )

    def _finish_requested(self) -> bool:
        if not (self.work_dir / "finish-search.request").is_file():
            return False
        if not self.report.session.get("finish_requested"):
            self.report.session["finish_requested"] = True
            self.log.append("search_finish_requested", reason="operator requested final validation")
            self.report.write(self.work_dir / "report.json")
        return True

    def _close_rule(self, run: RegionRun) -> str | None:
        """Spend the search budget, unless the operator requests finalization
        or nothing inside a kernel can pay any more: a launch-bound region
        whose head the last two attempts did not move, the last three kernels
        each within one sigma of the launch floor clocked beside them. One
        window is weak evidence: the floor probe swings about 30% between
        windows with the GPU's state, so three windows must agree."""
        if self._finish_requested():
            return "operator requested final validation"
        left = self._budget(run)
        if left["attempts_left_region"] <= 0:
            return "the region's hypothesis budget is spent"
        if left["attempts_left_job"] <= 0:
            return "the job budget is spent"
        roof = run.region.roofline
        if roof is not None and roof.bound == "launch" and run.head_age >= 2 and run.floor_streak >= 3:
            return ("no discernible headroom: the region is launch-bound, the last two attempts did "
                    "not move head, and the last three kernels each sat within one sigma of the "
                    "launch floor clocked beside them")
        return None

    # -- bind and promote -----------------------------------------------------

    def _emit_installation(self, definitions, name, *, replay=False):
        """Prefer graph insertion; retain certified replay for unsafe state boundaries."""
        from .bind.emit import covers_scope, scope_nodes
        if all(covers_scope(scope_nodes(trace, scope), [(c.start_seq, c.end_seq) for c in cuts])
               for trace, scope, cuts in definitions):
            # Replacing a complete calculation needs only a guarded kernel
            # call. There is no surrounding graph to rebuild or compile.
            self.log.append("installation", scope=definitions[0][1].address, method="direct")
            return emit_wrapper_variants(definitions, name)
        if not replay:
            from .bind.graph import emit_graph_wrapper_variants
            try:
                emitted = emit_graph_wrapper_variants(definitions, name)
                self.log.append("installation", scope=definitions[0][1].address, method="graph")
                return emitted
            except NotReplayable as error:
                self.log.append("graph_fallback", scope=definitions[0][1].address,
                                reason=str(error))
        self.log.append("installation", scope=definitions[0][1].address, method="replay")
        return emit_wrapper_variants(definitions, name)

    def _verify_installations(self, workload, tensors, states, cuts, baseline):
        """Inspect rewritten graphs directly; legacy replay keeps its literal retrace."""
        from autotuner_runtime.graph import GraphBindingError
        graph = {p: _resolve(self.model, p) for p in states
                 if isinstance(_resolve(self.model, p), GraphWrapper)}
        graph_spans = set()
        expected = {}
        for path, wrapper in graph.items():
            splices = [s for (w, _), ss in states[path][1].items() if w == workload for s in ss]
            graph_spans.update((s.start_seq, s.end_seq) for s in splices)
            expected[path] = Counter(s.kernel.kernel_id for s in splices)
        if graph:
            self.tracer.uninstall()

            def hits(calls):
                actual = Counter()
                for call in calls:
                    for row in call:
                        if row["hits"]:
                            actual[row["kernel_id"]] += row["hits"]
                return actual

            with ExitStack() as stack:
                observations = {p: stack.enter_context(wrapper.validate_graph()) for p, wrapper in graph.items()}
                try:
                    self.session.off_clock(lambda: self.model(*tensors))
                except GraphBindingError as error:
                    raise NotReplayable(str(error)) from error
                for path, calls in observations.items():
                    if hits(calls) != expected[path]:
                        raise NotReplayable(f"graph scope {path}: replaced {dict(hits(calls))}, expected {dict(expected[path])}")
            # The deployed path: a compiled scope's first call traces the same
            # substitutions, or runs the original and says why. Calls with one
            # signature share one trace, so evidence here is per signature and
            # each entry must have found its own rule's cuts.
            for wrapper in graph.values():
                wrapper._graph_evidence.clear()
            self.session.off_clock(lambda: self.model(*tensors))
            for path, wrapper in graph.items():
                reasons = wrapper._graph_fallbacks + [wrapper._graph_fallback_reason]
                if any(reasons):
                    raise NotReplayable(f"graph scope {path} ran the original when compiled: "
                                        + "; ".join(r for r in reasons if r))
                short = [row for call in wrapper._graph_evidence for row in call if row["hits"] != row["expected"]]
                if short:
                    raise NotReplayable(f"graph scope {path} compiled: {short}")
            self.log.append("graph_verified", workload=workload,
                            scopes={p: dict(v) for p, v in expected.items()})
        remaining = {span: kid for span, kid in cuts.items() if span not in graph_spans}
        if not graph or remaining:
            # Graph scopes are restored only for this check: the legacy log
            # describes Python calls, whereas graph insertion happens later.
            removed = []
            try:
                for path, wrapper in graph.items():
                    removed.append((path, swap_install(self.model, path, wrapper.wrapped)))
                self.tracer.uninstall()
                self.tracer.install()
                retrace, _ = self.tracer.trace(self.model, tensors)
                spans = sorted(remaining)
                result = verify_retrace(baseline, retrace, spans, [remaining[s] for s in spans])
                if not result.ok:
                    raise NotReplayable("; ".join(result.reasons))
            finally:
                self.tracer.uninstall()
                for path, wrapper in reversed(removed):
                    swap_install(self.model, path, wrapper)

    def _bind_and_promote(self, run: RegionRun, kernel: KernelSpec, result,
                          assoc_tag: str = "preserving") -> PromotionResult:
        """Bind the kernel into the live model and keep it only if the whole
        patched model is faster than what is already installed.

        The region clock nominates. A paired forward-pass comparison against
        the installed incumbent decides. Failed candidates are unwound."""
        if result.outcome != "tentative_ship" or kernel.reference_sequence is not None:
            return PromotionResult("not_tested", "region did not nominate an installable replacement")
        self.session.wait_ready()
        region = run.region
        installed_now: list[tuple[str, object]] = []
        emitted_before: dict[str, object] = {}   # scope -> prior entry (None = absent)
        pending_installed: dict[str, tuple] = {}
        pending_removed: set[str] = set()
        try:
            self.tracer.patcher.install()
        except RuntimeError:
            pass  # already installed

        def unwind():
            """Every failure path must leave the model, the artifact record,
            and the patch surface exactly as before this attempt."""
            failures = []
            for path, occupant in reversed(installed_now):
                try:
                    swap_uninstall(self.model, path, occupant)
                except Exception as e:
                    self.log.append("rollback_error", scope=path, reason=str(e))
                    failures.append(f"{path}: {e}")
            for path, prior in emitted_before.items():
                if prior is None:
                    self.emitted.pop(path, None)
                else:
                    self.emitted[path] = prior
            try:
                self.tracer.uninstall()
            except Exception as e:
                failures.append(f"tracer cleanup: {e}")
            if failures:
                raise RuntimeError("rollback failed; stopping before further GPU work: " + "; ".join(failures))

        baseline_traces = dict(self.traces)
        try:
            changes = {}
            for m in region.members:
                trace = self.traces[m.workload]
                scope = find_scope_call(trace, m.scope_stack)
                if scope is None:
                    raise NotReplayable(f"no scope call for {m.scope_stack!r}")
                scope_path = scope.address.rsplit("@", 1)[0]
                if scope_path == "":
                    raise NotReplayable("root-scope delivery is not supported yet")

                splice = Splice(
                    kernel=kernel, start_seq=m.start_seq, end_seq=m.end_seq,
                    input_ids=m.input_ids, output_ids=m.output_ids,
                    fingerprint=region.fingerprint,
                )
                changes.setdefault(scope_path, []).append((m.workload, scope, splice))

            # A parent replay bypasses child calls. Compose all their cuts at
            # the outermost scope, keeping each workload's trace-local ids.
            updates = compose_scope_variants(
                self.traces, {path: state[1] for path, state in self.installed.items()}, changes)
            identities = {}
            use_replay = set(self.replay_scopes)
            for scope_path, variants in updates.items():
                if scope_path in self.certified_scopes:
                    continue
                definitions = [(self.traces[w], next(sc for sc in self.traces[w].scope_calls
                                if sc.address == address), []) for w, address in variants]
                emitted_id = self._emit_installation(definitions, f"Id_{_safe(scope_path)}",
                                                     replay=scope_path in use_replay)
                identities[scope_path] = _load_class(emitted_id)(_resolve(self.baseline_model, scope_path), {})
            if identities:
                from autotuner_runtime.state import correctness_call
                from autotuner_runtime.graph import GraphBindingError
                from .bind.certify import CertificationResult
                self.log.append("identity_certification", phase="start", scopes=len(identities))
                try:
                    check = self.session.off_clock(lambda: certify_identities(
                        model=self.baseline_model, wrappers=identities,
                        runs=[lambda t=self.tensors[w.name]: correctness_call(self.baseline_model, t)
                              for w in self.manifest.workloads]), defer_cooling=True)
                except GraphBindingError as error:
                    check = CertificationResult(False, str(error))
                if check.ok:
                    # An identity that quietly ran the original proves nothing.
                    fell_back = [p for p, w in identities.items() if isinstance(w, GraphWrapper)
                                 and (w._graph_fallback_reason or w._graph_fallbacks)]
                    if fell_back:
                        check = CertificationResult(False, "graph identity ran the original module at "
                                                    + ", ".join(fell_back))
                self.log.append("identity_certification", phase="done", scopes=len(identities), passed=check.ok)
                if not check.ok:
                    # A scope whose identity changes under compilation can
                    # still use the previously validated replay mechanism.
                    failed_graph = {p for p, wrapper in identities.items() if isinstance(wrapper, GraphWrapper)}
                    if failed_graph:
                        use_replay.update(failed_graph)
                        self.replay_scopes.update(failed_graph)
                        self.log.append("graph_fallback", scopes=sorted(failed_graph), reason=check.reason)
                        for path in failed_graph:
                            definitions = [(self.traces[w], next(sc for sc in self.traces[w].scope_calls
                                            if sc.address == address), []) for w, address in updates[path]]
                            emitted_id = self._emit_installation(definitions, f"Id_{_safe(path)}", replay=True)
                            identities[path] = _load_class(emitted_id)(_resolve(self.baseline_model, path), {})
                        check = self.session.off_clock(lambda: certify_identities(
                            model=self.baseline_model, wrappers=identities,
                            runs=[lambda t=self.tensors[w.name]: correctness_call(self.baseline_model, t)
                                  for w in self.manifest.workloads]), defer_cooling=True)
                    if not check.ok:
                        self.log.append("certification_failed", scopes=list(identities), reason=check.reason)
                        unwind()
                        return PromotionResult("binding_failed", check.reason)
                self.certified_scopes.update(identities)
                for path in identities:
                    if not isinstance(identities[path], GraphWrapper):
                        for workload, _address in updates[path]:
                            region.delivery[workload] = "replay"
            # The incumbent at every scope this attempt compiles is that
            # scope compiled with the cuts it carried before, none on a first
            # install: compilation's own effect on the step never counts for
            # or against the kernel, whatever the scope absorbs.
            carried = compose_scope_variants(
                self.traces, {path: state[1] for path, state in self.installed.items()},
                {path: [] for path in changes})
            candidates = {}
            for scope_path, variants in updates.items():
                prior = self.installed.get(scope_path)
                original = prior[0] if prior else _resolve(self.model, scope_path)
                # The combined replay owns these kernels now. Restore the
                # underlying children so live fallback, clones and exported
                # wrappers all see the same original module tree.
                descendants = [path for path in self.installed
                               if path.startswith(scope_path + ".")]
                for path in sorted(descendants, key=lambda p: p.count("."), reverse=True):
                    occupant = swap_install(self.model, path, self.installed[path][0])
                    installed_now.append((path, occupant))
                    emitted_before.setdefault(path, self.emitted.get(path))
                    self.emitted.pop(path, None)
                    pending_removed.add(path)
                definitions = [
                    (self.traces[workload],
                     next(sc for sc in self.traces[workload].scope_calls if sc.address == address), cuts)
                    for (workload, address), cuts in variants.items()]
                kernels = {s.kernel.kernel_id: s.kernel for cuts in variants.values() for s in cuts}
                emitted = self._emit_installation(definitions, f"W_{_safe(scope_path)}",
                                                  replay=scope_path in use_replay)
                cls = _load_class(emitted)
                occupant = swap_install(self.model, scope_path, cls(original, kernels))
                installed_now.append((scope_path, occupant))
                emitted_before.setdefault(scope_path, self.emitted.get(scope_path))
                self.emitted[scope_path] = emitted
                pending_installed[scope_path] = (original, variants, kernels)
                # The incumbent at this scope: the same delivery carrying only
                # the cuts it carried before, so what the wrapper itself does
                # to the step (compiling it; a replay parent bypassing its
                # compiled children) never counts for or against the kernel.
                # A bare kernel call replaces a plain module and is measured
                # against it; a whole-model compiled baseline needs nothing.
                whole_model_compiled = self.baseline == "compiled" and not self.use_library_inference
                if not whole_model_compiled and "direct" not in region.delivery.values():
                    kept = {key: list(carried.get(scope_path, {}).get(key, [])) for key in variants}
                    prior_definitions = [
                        (self.traces[workload],
                         next(sc for sc in self.traces[workload].scope_calls if sc.address == address), cuts)
                        for (workload, address), cuts in kept.items()]
                    candidates[scope_path] = (
                        self._emit_installation(prior_definitions, f"Inc_{_safe(scope_path)}",
                                                replay=scope_path in use_replay),
                        {s.kernel.kernel_id: s.kernel for cuts in kept.values() for s in cuts})
            incumbent = self._copy_incumbent(
                {p: e for p, e in {**self.emitted, **emitted_before}.items() if e is not None}, candidates)
            compiled_scopes = sorted(p for p, (e, _) in candidates.items() if issubclass(_load_class(e), GraphWrapper))

            # retrace: every installed cut, this one included, must be exactly
            # one custom dispatch per copy against the job-start recording
            pending_cuts: dict[str, dict[tuple[int, int], str]] = {}
            affected_workloads = {workload for variants in updates.values()
                                  for workload, _address in variants}
            self.session.wait_ready()  # wrapper construction used the identity check's cooling time
            for w in self.manifest.workloads:
                new = {(m.start_seq, m.end_seq): kernel.kernel_id
                       for m in region.members if m.workload == w.name}
                if w.name not in affected_workloads:
                    continue
                cuts = {**self.cuts.get(w.name, {}), **new}
                states = {p: state for p, state in self.installed.items() if p not in pending_removed}
                states.update(pending_installed)
                self._verify_installations(w.name, self.tensors[w.name], states, cuts, baseline_traces[w.name])
                pending_cuts[w.name] = cuts

            self.tracer.uninstall()
            timed_arms = self._timed_arms(incumbent)
            e2e = run_e2e(
                self.session, self.baseline_model, self.model,
                workloads=[(w.name, self.tensors[w.name]) for w in self.manifest.workloads],
                timed=timed_arms,
                veto_pairs=SHIP_PAIRS,
                defer_cooling=True,
                exact=self._requires_exact(assoc_tag, region.fingerprint),
                tolerances=self.manifest.tolerances,
                target_workload=result.detail.get("target_workload"),
            )
            if not self._model_win(e2e):
                veto = e2e.veto
                self.log.append("not_shipped", fingerprint=region.fingerprint,
                                reason="outputs changed" if not all(c.passed for c in e2e.checks)
                                else "not faster",
                                model_verdicts={name: (
                                    "resolved_regression" if value.loses_by(0.0) else
                                    "resolved_improvement" if value.wins_by(0.0) else "inconclusive")
                                    for name, value in e2e.workload_vetos.items()},
                                checks=[c.__dict__ for c in e2e.checks],
                                veto={"baseline_ms": veto.median_baseline_ms,
                                      "delta_ms": veto.median_delta_ms, "sigma_ms": veto.sigma_ms,
                                      "ratio": veto.median_ratio, "stability": veto.stability}
                                if veto else None)
                unwind()
                status = ("correctness_failed" if not all(c.passed for c in e2e.checks) else
                          "regression" if any(v.loses_by(0.0) for v in e2e.workload_vetos.values()) else "inconclusive")
                return PromotionResult(status, "whole-model outputs changed" if status == "correctness_failed" else
                                       "whole-model regression" if status == "regression" else "whole-model speedup unresolved",
                                       [asdict(c) for c in e2e.checks],
                                       {name: asdict(v) for name, v in e2e.workload_vetos.items()})
            nomination_timings = {name: asdict(value) for name, value in e2e.workload_vetos.items()}
            e2e = self._confirm_model_win(e2e, incumbent, timed=timed_arms)
            if not self._model_win(e2e):
                self.log.append("not_shipped", fingerprint=region.fingerprint,
                                reason="whole-model win did not repeat in independent confirmation",
                                nomination_timings=nomination_timings,
                                confirmation_timings={name: asdict(value)
                                                      for name, value in e2e.workload_vetos.items()})
                unwind()
                return PromotionResult("confirmation_failed", "whole-model win did not repeat in independent confirmation",
                                       [asdict(c) for c in e2e.checks],
                                       {name: asdict(v) for name, v in e2e.workload_vetos.items()})
            model_ratios = {name: self.model_ratios[name] * value.median_ratio
                            for name, value in e2e.workload_vetos.items()}
            accepted = dict(
                fingerprint=region.fingerprint, kernel=kernel.kernel_id, assoc_tag=assoc_tag,
                target_workload=e2e.target_workload,
                model_ratios=model_ratios,
                model_ratio=next(iter(model_ratios.values())) if len(model_ratios) == 1 else None,
                model_ratio_kind="product_of_incremental_measurements",
                timing_baseline=("incumbent_with_compiled_scopes" if compiled_scopes else
                                 "previously_installed_model" if self.installed else "original_model"),
                incumbent_compiled_scopes=compiled_scopes,
                model_latency_reduction_pct={
                    name: 100 * (1 - value.median_ratio)
                    for name, value in e2e.workload_vetos.items()},
                checks=[asdict(c) for c in e2e.checks],
                timings={name: asdict(value) for name, value in e2e.workload_vetos.items()},
                nomination_timings=nomination_timings,
                confirmation="fresh_fixed_size_comparison",
            )
            # Write the exact accepted wrappers before committing the live
            # state. No GPU work: even a later device failure cannot lose them.
            accepted["checkpoint"] = str(self._save_checkpoint(accepted))
        except BaseException as e:
            unwind()
            from autotuner_runtime.graph import GraphBindingError
            if not isinstance(e, (NotReplayable, GraphBindingError)):
                raise  # interrupts, device errors and harness bugs must stop the job
            self.log.append("bind_failed", fingerprint=region.fingerprint,
                            reason=f"{type(e).__name__}: {e}")
            return PromotionResult("binding_failed", str(e))
        self.model_ratio = accepted["model_ratio"]
        self.model_ratios = accepted["model_ratios"]
        for path in pending_removed:
            self.installed.pop(path)
        self.installed.update(pending_installed)
        self.cuts.update(pending_cuts)
        self.shipped_tags[region.fingerprint] = assoc_tag
        self.report.accepted.append(accepted)
        self.log.append("shipped", **accepted)
        self.report.write(self.work_dir / "report.json")
        return PromotionResult("shipped", "whole-model correctness and independent timing confirmation passed",
                               accepted["checks"], accepted["timings"])

    def _save_checkpoint(self, accepted: dict) -> Path:
        """Durable recovery package; final timing and fresh-process export are pending."""
        path = self.work_dir / "checkpoints" / f"accepted-{len(self.report.accepted) + 1:04d}"
        snapshot = copy.deepcopy(self.report)
        snapshot.accepted.append({**accepted, "checkpoint": str(path)})
        snapshot.session.update(status="accepted_checkpoint", final_validation="pending",
                                artifact_validation="pending")
        specs = {}
        for scope in self.emitted:
            specs.update(getattr(_resolve(self.model, scope), "_specs", {}))
        bundle = self._model_bundle()
        # The live accepted-tag map is committed only AFTER this durable save.
        bundle.exact = self._requires_exact(accepted.get("assoc_tag"), accepted.get("fingerprint"))
        bundle.recovery = True  # final validation remains pending
        return emit_artifact(path, list(specs.values()), list(self.emitted.values()), snapshot,
                             bundle=bundle)

    def _model_win(self, e2e) -> bool:
        """Win on the target (or any final workload), with no resolved losses."""
        if not all(c.passed for c in e2e.checks) or e2e.veto is None:
            return False
        comparisons = e2e.workload_vetos
        if hasattr(self, "manifest") and set(comparisons) != {w.name for w in self.manifest.workloads}:
            return False  # early target rejection or incomplete timing is never a ship
        return workload_win(comparisons or {"main": e2e.veto}, getattr(e2e, "target_workload", None))

    def _confirm_model_win(self, nomination, incumbent, *, timed=None):
        """One independent, fixed-size confirmation, never retry-until-pass."""
        target = getattr(nomination, "target_workload", None)
        if target is None:
            target = next((name for name, value in sorted(getattr(nomination, "workload_vetos", {}).items())
                           if value.wins_by(0.0)), None)
        self.log.append("model_confirmation", phase="start", pairs=SHIP_PAIRS, target_workload=target)
        arms = timed if timed is not None else self._timed_arms(incumbent)
        timings = {}
        for name in sorted(arms, key=lambda name: (name != target, name)):
            timings[name] = compare(self.session, *arms[name], pairs=SHIP_PAIRS, defer_cooling=True)
            if name == target and not timings[name].wins_by(0.0):
                break
        result = E2EResult(checks=nomination.checks,
                          veto=next(iter(timings.values()), None), workload_vetos=timings,
                          target_workload=target)
        self.log.append("model_confirmation", phase="done", passed=self._model_win(result),
                        timings={name: asdict(value) for name, value in timings.items()})
        return result

    # -- the whole job --------------------------------------------------------

    def run(self) -> Report:
        """Keep a readable partial result even if preparation or search fails."""
        self.report.session["status"] = "running"
        self.report.write(self.work_dir / "report.json")
        try:
            report = self._run()
            report.session["status"] = "search_complete"
            return report
        except BaseException as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            detail = f"{type(error).__name__}: {error}"
            self.report.session.update(status=status, error=detail)
            self.log.append("job_failed", stage=self.report.session.get("stage"), reason=detail)
            raise
        finally:
            self.tracer.uninstall()
            self.report.write(self.work_dir / "report.json")

    def _phase(self, stage: str) -> None:
        self.report.session["stage"] = stage
        self.log.append("stage", name=stage)
        self.report.write(self.work_dir / "report.json")
        print(stage, flush=True)

    def _run(self) -> Report:
        self.log.append(
            "job", manifest=self.report.manifest_path,
            workloads={w.name: [[str(d) for d in i.shape] for i in w.inputs]
                       for w in self.manifest.workloads},
            budget_per_region=self.manifest.budget_per_region,
            budget_total=self.manifest.budget_total,
            defaulted=list(self.manifest.defaulted),
            gpu_utilization_pct=self.gpu_busy_at_start,
        )
        self._phase("loading model")
        self.load_model()
        self._phase("tracing workloads and finding legal regions")
        self.trace_workloads()
        regions = self.build_regions()
        self._phase("measuring baseline and selecting regions")
        ranked = self.capture_and_price(regions)
        self._phase("searching kernels")
        self.report.write(self.work_dir / "report.json")  # partial: peaks, coverage, strands

        shipped_regions: list[Region] = []
        while ranked or self.pending_regions:
            if self._finish_requested():
                break
            if self.total_hypotheses >= self.manifest.budget_total:
                break
            if not ranked:
                ranked = self._next_regions()
                if not ranked:
                    break
            region = ranked.pop(0)
            free = free_members(region, [r for r in shipped_regions if r is not region])
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
            accepted_before = len(self.report.accepted)
            roof = region.roofline
            self.log.append(
                "region_open", fingerprint=region.fingerprint, ops=list(region.ops),
                copies=region.copies, p=dict(region.p),
                bound=roof.bound if roof else None, s_max=roof.s_max if roof else None,
                roofline_ms=roof.t_roofline_ms if roof else None,
                t_orig_ms=dict(region.t_orig_ms), t_rep_ms=dict(region.t_rep_ms),
            )
            self.candidates.append(_region_line(region, "open"))
            judge = self.judge_factory(region)
            run = self.open_region(region, judge)
            if run.scaffold is not None and run.close_rule is None:
                self.hypothesis_cycle(run, judge)
            if run.shipped is not None and region not in shipped_regions:
                shipped_regions.append(region)
            self._record_region_closed(run)

            if self._finish_requested() or self.total_hypotheses >= self.manifest.budget_total:
                break
            if len(self.report.accepted) > accepted_before:
                ranked = self._refresh_after_ship(ranked, shipped_regions)

            # The completed region releases unshipped overlapping alternatives.
            # Remaining ranked targets block overlaps until their own search ends.
            ranked = rank(ranked + self._next_regions(blockers=ranked))

        selection = self.report.coverage.setdefault("selection", {})
        selection["unsearched"] = len(ranked) + len(self.pending_regions)

        # The headline is a ratio of two clocks taken together: the
        # untouched model is re-measured here, interleaved with the patched one,
        # never subtracted from the job-start clock taken on a different machine
        # state. "before" stays in the report as the job-start observation it is.
        self._phase("checking final model timings")
        for w in self.manifest.workloads if not self._kernels_installed() else []:
            tensors = self.tensors[w.name]
            comp = compare(
                self.session,
                self._step_fn(self.baseline_model, tensors),
                self._step_fn(self.model, tensors),
                pairs=self.clock_pairs,
            )
            self._record_final_clock(w.name, comp)

        self._final_check()
        self.report.session["idled_s"] = round(self.session.idled_s, 1)
        self.report.write(self.work_dir / "report.json")
        return self.report

    def _record_region_closed(self, run):
        self.report.add_region(
            fingerprint=run.region.fingerprint, ops=list(run.region.ops),
            copies=run.region.copies, workloads=list(run.region.workloads),
            p=dict(run.region.p), t_orig_ms=dict(run.region.t_orig_ms),
            t_rep_ms=dict(run.region.t_rep_ms),
            roofline_ms=run.region.roofline.t_roofline_ms if run.region.roofline else None,
            bound=run.region.roofline.bound if run.region.roofline else None, s_max=run.region.roofline.s_max if run.region.roofline else None,
            t_shipped_ms={run.shipped_workload: run.shipped_ms}
                if run.shipped_ms is not None and run.shipped_workload is not None else None,
            speedup=(1.0 / run.shipped_ratio) if run.shipped_ratio else None,
            close_rule=run.close_rule, hypotheses=run.hypotheses, head_ms=run.head_ms,
            timing_workload=run.shipped_workload,
            delivery=dict(run.region.delivery),
            library_arm={w: run.region.library_arm(w, self.baseline) for w in run.region.delivery},
        )
        self.log.append("region_closed", fingerprint=run.region.fingerprint,
                        rule=run.close_rule, shipped=run.shipped is not None,
                        hypotheses=run.hypotheses, head_ms=run.head_ms,
                        shipped_ms=run.shipped_ms,
                        head_workload=run.head_workload, shipped_workload=run.shipped_workload,
                        outcomes=self.report.regions[-1]["outcomes"])
        self.candidates.append(_region_line(run.region, "closed", run))
        self.report.write(self.work_dir / "report.json")  # a crash still leaves the story so far


    def _record_final_clock(self, workload, comp) -> None:
        after = statistics.median(comp.candidate_ms)
        confirmed = bool(self.installed) and comp.wins_by(0.0)
        clocks = {
            "after": after, "baseline_at_end": comp.median_baseline_ms,
            "speedup": (1.0 / comp.median_ratio) if comp.median_ratio else None,
            "stability": comp.stability, "win_confirmed": confirmed,
        }
        self.report.step_ms.setdefault(workload, {}).update(clocks)
        self.log.append("step_clock", workload=workload, phase="after", median_ms=after,
                        baseline_at_end_ms=comp.median_baseline_ms, speedup=clocks["speedup"],
                        stability=round(comp.stability, 3), win_confirmed=confirmed)

    def _final_check(self) -> None:
        """The last end-to-end check once the regions are done: the patched
        model as a whole against the untouched model, on every workload and
        then at every sweep size, where the wrappers must hand the unrecorded
        shapes back to the original modules."""
        self.session.wait_ready()
        if not self._kernels_installed():
            self.final_ok = True
            self.report.final = {"passed": True, "reason": "no replacements installed"}
            return
        final = run_e2e(
            self.session, self.baseline_model, self.model,
            workloads=[(w.name, self.tensors[w.name]) for w in self.manifest.workloads]
            + sorted(self.sweep_tensors.items()),
            timed=self._timed_arms(),
            veto_pairs=self.manifest.final_benchmark.pairs if self.use_library_inference else SHIP_PAIRS,
            exact=self._requires_exact(),
            tolerances=self.manifest.tolerances,
        )
        paired_passed = final.passed and self._model_win(final)
        for name, comp in final.workload_vetos.items():
            self._record_final_clock(name, comp)
        self.report.final = {"passed": paired_passed, "veto_passed": final.veto_passed,
                             "paired_win_confirmed": paired_passed,
                             "timings": {name: asdict(value) for name, value in final.workload_vetos.items()},
                             "checks": [c.__dict__ for c in final.checks]}
        if final.passed and self._compiled_baseline is None:
            self.report.final["delivery"] = self._final_delivery_split()
        self.log.append("final_e2e", passed=final.passed, veto_passed=final.veto_passed,
                        checks=self.report.final["checks"])
        if not final.passed:
            raise RuntimeError("final model validation failed; accepted checkpoints are preserved, "
                               "but correctness or the paired regression veto failed")
        if self.use_library_inference:
            # Each comparison already runs the full generation task, including
            # evolving state. Repeating it N times would change the objective.
            self.final_ok = paired_passed
            self.report.final["measurement"] = dict(self.report.constants["measurement"])
            self.report.write(self.work_dir / "report.json")
            if not self.final_ok:
                raise RuntimeError("final library inference timing did not confirm a speedup; "
                                   "accepted checkpoints are preserved")
            return
        # An inconclusive single-step clock cannot settle deployment performance.
        # Correct, non-regressing candidates still reach the consecutive-step test.
        # Only a confirmed sequence win permits export.
        self._phase("checking consecutive model steps")
        sequences, deployed = self._final_sequences(final.checks)
        self.report.final["sequences"] = sequences
        # the same decision rule as every install: faster with confidence,
        # now over whole uninterrupted runs of the model
        self.final_ok = self._model_win(deployed)
        self.report.final["passed"] = self.final_ok
        for name, row in sequences.items():
            self.report.step_ms[name]["sequence_win_confirmed"] = row["win_confirmed"]
            self.report.step_ms[name]["sequence_speedup"] = row["speedup"]
            self.report.step_ms[name]["win_confirmed"] &= self.final_ok
        self.report.write(self.work_dir / "report.json")
        if not self.final_ok:
            raise RuntimeError("consecutive-step validation did not confirm a correct speedup; "
                               "see final.sequences in report.json; accepted checkpoints are preserved")

    def _final_delivery_split(self) -> dict:
        """Split the final speedup: the untouched model against the installed
        scopes compiled with no kernel, and that against the patched model.
        Only a plain baseline with graph-delivered scopes has anything to split."""
        if self.baseline != "plain":
            return {"compiled_scopes": [], "timings": {}}
        compiled = {}
        for path in self.emitted:
            if isinstance(_resolve(self.model, path), GraphWrapper):
                compiled[path] = (self._identity(path, f"Id_{_safe(path)}"), {})
        if not compiled:
            return {"compiled_scopes": [], "timings": {}}
        identity = self._copy_incumbent({}, compiled)
        pairs = self.manifest.final_benchmark.pairs if self.use_library_inference else SHIP_PAIRS
        timings, arms = {}, self._timed_arms()
        for w in self.manifest.workloads:
            plain, patched = arms[w.name]
            identity_step = self._step_fn(identity, self.tensors[w.name])
            timings[w.name] = {
                "plain_vs_compiled_identity": asdict(compare(self.session, plain, identity_step, pairs=pairs)),
                "compiled_identity_vs_patched": asdict(compare(self.session, identity_step, patched, pairs=pairs)),
            }
        split = {"compiled_scopes": sorted(compiled), "timings": timings}
        self.log.append("final_delivery_split", **split)
        return split

    def _final_sequences(self, checks) -> tuple[dict, E2EResult]:
        """Time the model the way it is deployed: whole runs of consecutive
        steps, original and patched, each run uninterrupted, the two
        alternated in both orders with cooling between runs. Returns the
        report rows and the comparison in the form every install decision
        reads."""
        config = self.manifest.final_benchmark
        results, timings = {}, {}
        sequence_checks = list(checks)
        for workload in self.manifest.workloads:
            name, tensors = workload.name, self.tensors[workload.name]
            steps = []
            for model in (self.baseline_model, self.model):
                # compile the forward call only, as a caller would
                call = lambda *args, m=model: m(*args)
                steps.append(mx.compile(call) if self.baseline == "compiled" else call)
            runs = [make_sequence(step, tensors, config.steps) for step in steps]
            kind = "repeated_forward"
            if self.use_library_inference:
                kind = "library_generation"
                runs = [lambda m=model: m(*tensors) for model in (self.baseline_model, self.model)]
            elif self.context is not None:
                kind = "advancing_cache_fixed_tokens"
                models = [ContextSequence(model, config.steps)
                          for model in (self.baseline_model, self.model)]
                runs = [lambda m=model: m(*tensors) for model in models]
            from autotuner_runtime.state import sequence_observation
            correctness_steps = ((self.baseline_model, self.model) if self.context is not None else steps)
            correctness_runs = [lambda step=step: sequence_observation(step, tensors, config.steps)
                                for step in correctness_steps]
            check = self.session.off_clock(
                lambda: preserving_check(*correctness_runs, name + ":sequence", exact=self._requires_exact(),
                                         tolerances=self.manifest.tolerances))
            sequence_checks.append(check)
            if not check.passed:
                raise RuntimeError(f"consecutive model outputs failed for {name}: {check.reason}")
            warm = [lambda s=step: s(*tensors) for step in steps]
            row = compare_sequences(self.session, *runs, *warm, pairs=config.pairs,
                                    warmup_steps=config.warmup_steps, label=name)
            timings[name] = comparison_from_samples(row["timing"]["baseline_ms"],
                                                    row["timing"]["candidate_ms"])
            row["steps"] = config.steps
            row["workload_kind"] = kind
            row["prefix_copy_included"] = self.context is not None
            results[name] = row
            self.log.append("sequence_comparison", workload=name, **row)
            self.report.final.setdefault("sequences", {})[name] = row
            self.report.write(self.work_dir / "report.json")
        deployed = E2EResult(checks=sequence_checks, veto=next(iter(timings.values()), None),
                             workload_vetos=timings)
        return results, deployed

    def emit_artifact(self, out_dir: str | Path) -> Path:
        self._phase("exporting and checking artifact")
        try:
            out = self._emit_artifact(out_dir)
            self.report.session.update(status="complete", artifact=str(out))
            self.report.write(self.work_dir / "report.json")
            self.report.write(out / "report.json")
            return out
        except BaseException as error:
            self.report.session.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=f"{type(error).__name__}: {error}")
            self.log.append("job_failed", stage="exporting and checking artifact",
                            reason=self.report.session["error"])
            self.report.write(self.work_dir / "report.json")
            raise

    def _requires_exact(self, pending_tag: str | None = None, pending_region: str | None = None):
        """Numerical policy follows installed edits, including a pending replacement."""
        return pending_tag != "changing" and not any(
            tag == "changing" for region, tag in self.shipped_tags.items()
            if pending_region is None or region != pending_region)

    def _model_bundle(self) -> ModelBundle:
        """The model source, traced inputs and final-check settings the artifact
        carries so it loads, validates and re-times without the harness."""
        context = None if self.context is None else {
            "workload": self.context.name, "context": self.context.context,
            "seed": self.context_seed, "tokens": self.context_tokens}
        return ModelBundle(self.manifest.model_path, self.baseline,
                           {**self.tensors, **self.sweep_tensors}, self.manifest.workloads,
                           asdict(self.manifest.final_benchmark), context=context,
                           exact=self._requires_exact(), tolerances=self.manifest.tolerances,
                           use_library_inference=bool(self.use_library_inference),
                           checkpoint_pins=getattr(self, "checkpoint_pins", None))

    def _emit_artifact(self, out_dir: str | Path) -> Path:
        """Write the artifact, then load it onto a fresh build() in a fresh
        process and check it reproduces the patched model."""
        if not self.final_ok:
            raise RuntimeError("the final whole-model check failed; no artifact is written "
                               "for a patched model that does not match the original")
        specs: dict[str, KernelSpec] = {}
        for path in self.emitted:
            wrapper = _resolve(self.model, path)
            specs.update(getattr(wrapper, "_specs", {}))
        # Fresh export comparisons build their own original with shared weights.
        cases = [(name, tensors, []) for name, tensors in
                 list(self.tensors.items()) + sorted(self.sweep_tensors.items())]
        context = None if self.context is None else (
            self.context.context, self.context_tokens, self.tensors[self.context.name])
        self.session.wait_ready()
        out = emit_artifact(
            out_dir, list(specs.values()), list(self.emitted.values()), self.report,
            validate=lambda staged: check_apply_many(staged, self.manifest.model_path, cases,
                                                     context=context),
            bundle=self._model_bundle())
        self.log.append("artifact_checked", artifact=str(out), workloads=[c[0] for c in cases])
        return out


def _state_marks(trace: Trace) -> str:
    """Why this step cannot be compiled from outside the model, or "" if it
    can: every recorded sign that the step is not a pure function of its
    inputs. An array the model kept after the pass, a call that changed an
    object's state (a KV cache write), or an evaluation mid-step (a branch on
    a value): mx.compile breaks on each."""
    marks = []
    if arrays := len(trace.python_retained()):
        marks.append(f"{arrays} arrays kept")
    if calls := len(trace.state_calls()):
        marks.append(f"{calls} state calls")
    if trace.in_pass_evaluation:
        marks.append("evaluates mid-step")
    return ", ".join(marks)


def _safe(path: str) -> str:
    return path.replace(".", "_") or "root"


def _ops_view(trace: Trace, stretch: Stretch) -> list[dict]:
    """The recorded calls of one copy, with arrays named by role (in0, out0,
    t<seq> for intermediates) and every non-tensor argument spelled out."""
    names = {aid: f"in{i}" for i, aid in enumerate(stretch.input_ids)}
    names.update({aid: f"out{i}" for i, aid in enumerate(stretch.output_ids)})
    view = []
    for node in trace.nodes[stretch.start_seq:stretch.end_seq + 1]:
        for output, aid in enumerate(node.out_arrays):
            names.setdefault(aid, f"t{node.seq}_{output}")

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
    from .judge.prompts import diagnostic_metadata
    return diagnostic_metadata(view)


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



def _kernel_id(region: Region, tag: str) -> str:
    """Full region identity keeps globally stored kernels distinct."""
    return f"r{region.fingerprint}_{tag}"


def resolve_parent(run: RegionRun, name: str) -> KernelSpec | None:
    """The kernel a proposal edits: head, scaffold, shipped, a hypothesis id,
    or a kernel id; None when the name matches nothing in this region."""
    by_role = {"head": run.head, "scaffold": run.scaffold, "shipped": run.shipped}
    if name in by_role:
        return by_role[name]
    if name in run.kernels:
        return run.kernels[name]
    return run.kernels.get(_kernel_id(run.region, name))


def kernel_from_proposal(contract: RegionContract, parent: KernelSpec | None, proposal,
                         kernel_id: str) -> KernelSpec:
    """The harness owns names, dtypes, and the call site; the judge supplies
    the body and the launch. A header or template left out means the parent's."""
    scratch = tuple(proposal.scratch)
    header = proposal.header or (parent.header if parent else "")
    native = parent.native_call if parent else None
    stages = tuple(replace(stage, kernel=replace(stage.kernel,
                   kernel_id=f"{kernel_id}_stage{i}", name=f"at_{kernel_id}_stage{i}",
                   header=stage.kernel.header or header))
                   for i, stage in enumerate(proposal.stages))
    atomic_outputs = False
    if parent is not None and not stages:
        atomic_outputs = native["factory"].get("atomic_outputs", False) if native else parent.atomic_outputs
    return KernelSpec(
        kernel_id=kernel_id,
        name=f"at_{kernel_id}",
        input_names=contract.input_names,
        output_names=contract.output_names + tuple(s[0] for s in scratch),
        source=proposal.source,
        header=header,
        grid=tuple(proposal.grid),
        threadgroup=tuple(proposal.threadgroup),
        output_shapes=(tuple(tuple(s) for s in proposal.output_shapes)
                       + tuple(tuple(s[2]) for s in scratch)),
        output_dtypes=contract.output_dtypes + tuple(s[1] for s in scratch),
        template=() if stages else tuple(proposal.template) or (parent.template if parent else ()),
        fallback_predicate=proposal.fallback_predicate,
        native_call=native,
        input_signature=contract.input_signature,
        ensure_row_contiguous=parent.ensure_row_contiguous if parent else True,
        atomic_outputs=atomic_outputs,
        stages=stages,
    )


def _kernel_view(spec: KernelSpec) -> dict:
    """What the judge may see of a kernel: its source and launch story.
    Native call-site settings are visible as read-only context, never proposal fields."""
    from .judge.prompts import diagnostic_metadata
    return {
        "native_call": diagnostic_metadata(spec.native_call),
        "input_signature": spec.input_signature,
        "reference_sequence": diagnostic_metadata(spec.reference_sequence),
        "source": spec.source,
        "header": spec.header,
        "grid": list(spec.grid),
        "threadgroup": list(spec.threadgroup),
        "output_shapes": [list(s) for s in spec.output_shapes],
        "output_dtypes": list(spec.output_dtypes),
        "template": [list(t) for t in spec.template],
        "stages": [{"inputs": list(stage.inputs), "outputs": list(stage.outputs),
                    **_kernel_view(stage.kernel)} for stage in spec.stages],
        "input_names": list(spec.input_names),
        "output_names": list(spec.output_names),
    }


# the acceptance envelope: values the judge could tune an edit to sit under
def _safe_detail(detail: dict | None) -> dict:
    """Judge-visible gate detail: distances like max_excess only, never the
    acceptance envelope. The judge may learn how far a check missed, never
    where the line is."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items()
                    if not isinstance(k, str) or k.lower() not in
                    ENVELOPE_KEYS | {"reference", "candidate"}}
        if isinstance(value, (list, tuple)):
            return [strip(v) for v in value]
        return value
    from .judge.prompts import diagnostic_metadata
    return diagnostic_metadata(strip(detail)) if detail else {}


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
