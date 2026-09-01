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
from .e2e import run_e2e, share_weights
from .judge.queue import FamilyBook, Queue, QueueError
from .judge.schema import JudgeBabble
from .ladder.gates import EvalSet, LadderJob, LadderResult, run_ladder
from .ladder.static_checks import RegionContract
from .artifact.emit import write_kernel
from .log import RunLog, TextLog, wall_now
from .measure.clocks import compare, step_clock
from .measure.controls import aa_null
from .measure.peaks import implausible as peaks_implausible, measure_peaks
from .measure.session import Session
from .regions.build import build_stretches
from .regions.fingerprint import group_copies
from .regions.price import capture_boundaries, price_region
from .regions.rank import apply_floor, covered_by, rank
from .regions.roofline import stretch_roofline
from .regions.store import BoundaryStore
from .regions.types import Region, Stretch
from .report import Report
from .trace import Tracer
from .trace.serialize import nodes_to_json
from .trace.types import Trace
from .workload import materialize, workload_seeds
from autotuner_runtime.kernels import KernelSpec

MIN_WIN_MS = 0.030  # plan 5.12: a few tens of microseconds per step across copies
CLOCK_PAIRS = 32    # ABBA pairs behind the ship clock and the headline
STALE_LIMIT = 6
FAIL_STREAK_LIMIT = 5
SHIP_ROOFLINE_CLOSE = 1.05
DIMINISHING_SHIPS = 3
DIMINISHING_PCT = 0.02


@dataclass
class RegionRun:
    region: Region
    scaffold: KernelSpec | None = None
    head: KernelSpec | None = None
    shipped: KernelSpec | None = None
    head_ms: float | None = None
    shipped_ms: float | None = None
    hypotheses: int = 0
    stale_streak: int = 0
    recent_ship_gains: list[float] = field(default_factory=list)
    fail_streak_by_parent: dict[str, int] = field(default_factory=dict)
    close_rule: str | None = None
    kernels: dict[str, KernelSpec] = field(default_factory=dict)


class JobRunner:
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
        self.session = session or Session(log_path=self.work_dir / "session.jsonl")
        self.log = RunLog(self.work_dir / "run.jsonl")
        self.candidates = TextLog(self.work_dir / "candidates.log")
        self.kernel_dir = self.work_dir / "kernels"
        self.report = Report(manifest_path=str(manifest_path))
        self.store = BoundaryStore(self.work_dir / "boundaries")
        self.tracer = Tracer()
        self.traces: dict[str, Trace] = {}
        self.tensors: dict[str, list[mx.array]] = {}
        self.step_ms: dict[str, float] = {}
        self.total_hypotheses = 0
        self.installed: dict[str, tuple] = {}  # scope -> (original module, splices, kernels)
        self.certified_scopes: set[str] = set()
        self.emitted: dict[str, EmittedWrapper] = {}

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
        self.log.append("model", shared_weights=shared)

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

    # -- stage 2: regions -----------------------------------------------------

    def build_regions(self) -> list[Region]:
        stretches = {name: build_stretches(t, name) for name, t in self.traces.items()}
        regions = group_copies(self.traces, stretches)
        viable = []
        for r in regions:
            reason = screen_scope(self.traces[r.members[0].workload], r.members[0].scope_stack)
            if reason is not None:
                r.rejected = f"no certified delivery scope: {reason}"
                self.report.stranded.append({"fingerprint": r.fingerprint, "ops": list(r.ops),
                                             "reason": r.rejected})
            else:
                viable.append(r)
        self.log.append("regions", candidates=len(regions), screened=len(viable))
        return viable

    def capture_and_price(self, regions: list[Region]) -> list[Region]:
        # capture: k input sets per workload, boundaries of every viable region
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

        # the recorder must be fully removed before any timing (law: pass 2)
        self.tracer.uninstall()
        mx.clear_cache()  # capture's activation buffers; warm-up refills what clocks need
        for w in self.manifest.workloads:
            tensors = self.tensors[w.name]
            clock = step_clock(self.session, lambda t=tensors: self.model(*t))
            self.step_ms[w.name] = clock.median_ms
            self.report.step_ms[w.name] = {"before": clock.median_ms}
            self.log.append("step_clock", workload=w.name, phase="before",
                            median_ms=clock.median_ms)
        self.peaks = measure_peaks(self.session)
        self.report.peaks = {"bandwidth_gbps": self.peaks.bandwidth_gbps,
                             "flops_gflops": self.peaks.flops_gflops,
                             "launch_us": self.peaks.launch_us}
        floor = aa_null(self.session, pairs=8)
        self.report.session["aa_floor_sigma_ms"] = floor.sigma_ms
        self.log.append("peaks", **self.report.peaks, aa_floor_sigma_ms=floor.sigma_ms)
        warn = peaks_implausible(self.peaks)
        if warn:
            # comparisons stay valid (paired), but the machine is degraded and
            # every absolute number in this job is suspect; say so loudly
            self.log.append("env_warning", detail=warn)
            print(f"WARNING: {warn}; this machine is degraded, absolute clocks "
                  "and rooflines from this job are not representative")

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
            step_fns = {w.name: (lambda t=self.tensors[w.name]: self.model(*t))
                        for w in self.manifest.workloads}
            price_region(r, self.session, self.traces, sets_for, weights, step_fns,
                         pairs=self.clock_pairs)
            rep = r.members[0]
            # One-copy cost against the one-copy floor; the all-copies total
            # would inflate s_max by the copy count. The cost is the measured
            # share re-expressed in the frame the peaks were taken in, so both
            # sides of s_max come from one machine state: priced while the chip
            # is throttled and divided by a healthy peak, a region at its
            # roofline reports headroom it does not have.
            share = r.p_rep.get(rep.workload)
            r.roofline = stretch_roofline(
                self.traces[rep.workload], rep, self.peaks,
                t_orig_ms=(share * self.step_ms[rep.workload]) if share else 1e-9,
            )
        kept = rank(apply_floor(regions))
        covered_ms = sum(sum(r.t_orig_ms.values()) for r in regions)
        self.report.coverage = {
            "sum_region_clock_x_copies_ms": covered_ms,
            "step_ms": dict(self.step_ms),
        }
        for r in regions:
            if r.rejected:  # share floor and no-headroom cuts both belong in the report
                self.report.stranded.append({"fingerprint": r.fingerprint, "ops": list(r.ops),
                                             "reason": r.rejected})
        self.log.append("ranked", kept=len(kept))
        return kept

    # -- stage 3: one region --------------------------------------------------

    def _contract(self, region: Region) -> RegionContract:
        rep = region.members[0]
        trace = self.traces[rep.workload]
        specs = {}
        for n in trace.nodes[rep.start_seq:rep.end_seq + 1]:
            for aid, s in zip(n.in_arrays, n.in_specs):
                specs.setdefault(aid, s)
            for aid, s in zip(n.out_arrays, n.out_specs):
                specs[aid] = s
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
                # ladder validation as a bare None (audit finding 4)
                raise RuntimeError(f"region eval set unpriced for workload {m.workload!r}")
            sets.append(EvalSet(
                label=m.workload,
                inputs_paths=[str(self.store._path(region.fingerprint, m.workload, i, "inputs"))
                              for i in range(k)],
                reference_paths=[str(self.store._path(region.fingerprint, m.workload, i, "outputs"))
                                 for i in range(k)],
                t_library_ms=region.t_orig_ms.get(m.workload),
                correctness_only=False,
                nodes_json=None,
            ))
        return sets

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
        return LadderJob(
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
            timeout_s=120.0,
        )

    def _kernel_from_proposal(self, run: RegionRun, region: Region, proposal, hyp_id: str) -> KernelSpec:
        kid = f"r{region.fingerprint[:6]}_{hyp_id}"
        return kernel_from_proposal(self._contract(region), run.kernels, proposal, kid)

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
            scaffold = build_scaffold(trace, rep)
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
        write_kernel(self.kernel_dir, scaffold)
        result = run_ladder(self._ladder_job(region, scaffold, "preserving", run_clock=False))
        self._record_attempt(region, "scaffold", "scaffold", "the harness's starting kernel",
                             "preserving", scaffold, None, result)
        if result.outcome == "failed":
            self.log.append("scaffold_failed", fingerprint=region.fingerprint,
                            gate=result.failed_gate, detail=result.detail)
            # one judge fix attempt, per the spec
            fixed = self._judge_fix(run, judge, scaffold, result)
            if fixed is None:
                run.close_rule = f"scaffold failed {result.failed_gate} (the one judge fix attempt did not produce a passing kernel)"
                return run
            scaffold = fixed
        run.scaffold = run.head = scaffold
        run.kernels[scaffold.kernel_id] = scaffold
        self.log.append("scaffold_ok", fingerprint=region.fingerprint,
                        kernel=scaffold.kernel_id)
        return run

    def _rename(self, spec: KernelSpec, region: Region, tag: str) -> KernelSpec:
        kid = f"r{region.fingerprint[:6]}_{tag}"
        d = {k: getattr(spec, k) for k in spec.__dataclass_fields__}
        d["kernel_id"] = kid
        d["name"] = f"at_{kid}"
        return KernelSpec(**d)

    def _judge_fix(self, run: RegionRun, judge, scaffold, result) -> KernelSpec | None:
        region = run.region
        # the judge needs the full region state plus the failing kernel to fix it
        run.kernels[scaffold.kernel_id] = scaffold
        meta = self._meta(run)
        meta["scaffold_failure"] = True
        meta["head"] = scaffold.kernel_id
        meta["kernels"] = {scaffold.kernel_id: _kernel_view(scaffold)}
        try:
            resp = judge.next(meta, {"outcome": "failed", "failed_gate": result.failed_gate,
                                     "detail": _safe_detail(result.detail)})
        except Exception as e:
            self.log.append("scaffold_fix", fingerprint=region.fingerprint,
                            outcome="judge_error", reason=str(e)[:200])
            return None
        if resp.kernel is None:
            self.log.append("scaffold_fix", fingerprint=region.fingerprint, outcome="yield")
            return None
        fixed = self._kernel_from_proposal(run, region, resp.kernel, "scafix")
        write_kernel(self.kernel_dir, fixed)
        check = run_ladder(self._ladder_job(region, fixed, "preserving", run_clock=False))
        self._record_attempt(region, "scafix", "fix", "the judge's one fix of the starting kernel",
                             "preserving", fixed, resp.kernel.parent_kernel_id, check)
        if check.outcome == "failed":
            self.log.append("scaffold_fix", fingerprint=region.fingerprint,
                            outcome="fix_failed_ladder", gate=check.failed_gate,
                            detail=check.detail)
            return None
        self.log.append("scaffold_fix", fingerprint=region.fingerprint, outcome="fixed")
        return fixed

    def hypothesis_cycle(self, run: RegionRun, judge) -> None:
        region = run.region
        queue = Queue()
        families = FamilyBook()
        families.register_scaffold(run.scaffold.kernel_id)
        t0 = time.perf_counter()
        try:
            seed = judge.seed(self._meta(run))
        except JudgeBabble:
            self.log.append("judge", fingerprint=region.fingerprint, phase="seed",
                            latency_s=round(time.perf_counter() - t0, 1), action="babble")
            run.close_rule = "queue empty (judge babbled at seed)"
            return
        except Exception as e:
            # a transport blip (CLI exit, timeout, network) costs the region,
            # never the job (audit finding F1)
            self.log.append("judge", fingerprint=region.fingerprint, phase="seed",
                            latency_s=round(time.perf_counter() - t0, 1),
                            action="error", reason=str(e)[:200])
            run.close_rule = f"judge unavailable at seed ({type(e).__name__})"
            return
        self.log.append("judge", fingerprint=region.fingerprint, phase="seed",
                        latency_s=round(time.perf_counter() - t0, 1), queue=len(seed.queue))
        try:
            queue.seed(seed.queue)
        except QueueError as e:
            run.close_rule = f"the judge's seed queue was inconsistent ({e})"
            return
        verdict_payload = None
        judge_errors = 0  # consecutive transport failures; 3 closes the region

        while True:
            rule = self._close_rule(run, queue)
            if rule:
                run.close_rule = rule
                return
            item = queue.pop_ready()
            if item is None:
                if queue.empty:
                    run.close_rule = "the queue is empty (the judge yields)"
                else:
                    run.close_rule = (f"no queued item is ready; {len(queue)} wait on "
                                      f"unsatisfied conditions ({', '.join(queue.ids())})")
                return
            # the queue and verdict log are the judge's only memory (spec);
            # every next call carries them plus the item it must write for
            meta = self._meta(run)
            meta["queue"] = list(queue.snapshot())
            meta["verdicts"] = queue.verdicts
            meta["families"] = families.state()
            meta["executing"] = {"id": item.id, "kind": item.kind,
                                 "assoc_tag": item.assoc_tag,
                                 "hypothesis": item.hypothesis}
            t0 = time.perf_counter()
            try:
                resp = judge.next(meta, verdict_payload)
            except JudgeBabble:
                self.log.append("judge", fingerprint=region.fingerprint, phase="next",
                                latency_s=round(time.perf_counter() - t0, 1), action="babble")
                run.hypotheses += 1
                self.total_hypotheses += 1
                self._record_attempt(region, item.id, item.kind, item.hypothesis,
                                     item.assoc_tag, None, None, None, gate="judge_babble")
                verdict_payload = {"outcome": "failed", "failed_gate": "judge_babble"}
                queue.record_verdict(item.id, "failed")
                continue
            except Exception as e:
                self.log.append("judge", fingerprint=region.fingerprint, phase="next",
                                latency_s=round(time.perf_counter() - t0, 1),
                                action="error", reason=str(e)[:200])
                judge_errors += 1
                self._record_attempt(region, item.id, item.kind, item.hypothesis,
                                     item.assoc_tag, None, None, None, gate="judge_error",
                                     reason=str(e)[:200])
                verdict_payload = {"outcome": "failed", "failed_gate": "judge_error",
                                   "detail": {"reason": str(e)[:200]}}
                queue.record_verdict(item.id, "failed")
                if judge_errors >= 3:
                    run.close_rule = f"judge unavailable (3 straight transport errors, last {type(e).__name__})"
                    return
                continue
            judge_errors = 0
            self.log.append("judge", fingerprint=region.fingerprint, phase="next",
                            latency_s=round(time.perf_counter() - t0, 1),
                            mutations=len(resp.mutations),
                            action="yield" if resp.kernel is None else "proposal")
            try:
                queue.apply_mutations(resp.mutations)
            except QueueError as e:
                # a bad plan edit costs the plan edit, never the kernel or job
                self.log.append("queue_mutations_rejected",
                                fingerprint=region.fingerprint, reason=str(e))
            if resp.kernel is None:
                run.close_rule = "the queue is empty (the judge yields)"
                return

            run.hypotheses += 1
            self.total_hypotheses += 1
            parent = resp.kernel.parent_kernel_id
            known = parent in run.kernels
            kernel = self._kernel_from_proposal(run, region, resp.kernel, item.id)
            run.kernels[kernel.kernel_id] = kernel
            write_kernel(self.kernel_dir, kernel)
            family = families.resolve(item, parent)
            # a child joins its parent's family; a fresh one would never trip the
            # eight-strike abandonment rule
            families.register_kernel(kernel.kernel_id, family)
            if known:
                result = run_ladder(self._ladder_job(region, kernel, item.assoc_tag, run_clock=True))
            else:
                # the parent is the judge's own memory of what it edited; an
                # unknown one is a mistake to name, not a kernel to measure
                result = LadderResult("failed", "static", {"failures": [{
                    "check": "parent_kernel_id",
                    "detail": f"{parent!r} is not a kernel of this region; "
                              f"the kernels are {sorted(run.kernels)}"}]},
                    None, None, None, None, [])
            if result.outcome == "failed":
                outcome = "failed"
                if result.failed_gate in ("static", "compile"):
                    run.fail_streak_by_parent[parent] = run.fail_streak_by_parent.get(parent, 0) + 1
                run.stale_streak += 1
            else:
                run.fail_streak_by_parent[parent] = 0
                if result.outcome == "correct_slower":
                    outcome = "correct_slower"
                    run.head, run.head_ms = kernel, result.region_ms
                    run.stale_streak += 1
                else:  # tentative_ship: bind and e2e decide
                    if self._bind_and_promote(run, kernel, result):
                        outcome = "shipped"
                        gain = result.win_ms or 0.0
                        prev = run.shipped_ms
                        run.shipped, run.shipped_ms = kernel, result.region_ms
                        run.head, run.head_ms = kernel, result.region_ms
                        run.recent_ship_gains.append(
                            gain / prev if prev else 1.0
                        )
                        run.stale_streak = 0
                    else:
                        outcome = "rolled_back"
                        run.stale_streak += 1

            families.record_verdict(family, outcome if outcome != "rolled_back" else "rolled_back")
            if families.tripped(family):
                families.abandon(family)
                run.head = run.shipped or run.scaffold
                self.log.append("family_abandoned", fingerprint=region.fingerprint, family=family)
            queue.record_verdict(item.id, outcome)
            verdict_payload = {
                "outcome": outcome,
                "failed_gate": result.failed_gate,
                "detail": _safe_detail(result.detail),
                "region_ms": result.region_ms,
            }
            self._record_attempt(region, item.id, item.kind, item.hypothesis, item.assoc_tag,
                                 kernel, parent, result, outcome=outcome)

    def _record_attempt(self, region: Region, hyp_id: str, kind: str, text: str,
                        assoc_tag: str | None, kernel, parent: str | None, result,
                        outcome: str | None = None, gate: str | None = None,
                        reason: str | None = None) -> None:
        """One attempt, written three ways: the report row, the run.jsonl
        verdict row, and one line of candidates.log."""
        if result is None:  # the judge produced nothing to evaluate
            outcome, detail = "failed", {"reason": reason} if reason else {}
            region_ms = library_ms = win_ms = sigma_ms = None
        else:
            outcome = outcome or result.outcome
            gate, detail = result.failed_gate, result.detail
            region_ms, library_ms = result.region_ms, result.library_ms
            win_ms, sigma_ms = result.win_ms, result.sigma_ms
        self.report.add_hypothesis(
            hypothesis_id=hyp_id, region=region.fingerprint, kind=kind,
            hypothesis_text=text, assoc_tag=assoc_tag, parent=parent,
            kernel=kernel.kernel_id if kernel else None, verdict=outcome,
            failed_gate=gate, region_ms=region_ms, library_ms=library_ms,
            win_ms=win_ms, sigma_ms=sigma_ms,
        )
        self.log.append("verdict", fingerprint=region.fingerprint, hypothesis=hyp_id,
                        hypothesis_kind=kind, hypothesis_text=text, assoc_tag=assoc_tag,
                        kernel=kernel.kernel_id if kernel else None, parent=parent,
                        outcome=outcome, gate=gate, region_ms=region_ms,
                        library_ms=library_ms, win_ms=win_ms, sigma_ms=sigma_ms,
                        detail=detail)
        if gate:
            how = f"{outcome} at {gate}: {_short_reason(detail)}"
        elif region_ms is not None and library_ms is not None:
            how = (f"{outcome}: {region_ms:.4f} ms vs library {library_ms:.4f} ms per copy "
                   f"(win {win_ms:+.4f}, sigma {sigma_ms:.4f})")
        elif outcome == "correct_slower":
            how = "correct, not clocked"
        else:
            how = outcome
        self.candidates.append(
            f"{wall_now()}\t{region.fingerprint[:8]}\t{hyp_id}\t{kind}\t{how}\t{text}")

    def _meta(self, run: RegionRun) -> dict:
        region = run.region
        rep = region.members[0]
        trace = self.traces[rep.workload]
        specs = {}
        for n in trace.nodes[rep.start_seq:rep.end_seq + 1]:
            for aid, s in zip(n.in_arrays, n.in_specs):
                specs.setdefault(aid, s)
            for aid, s in zip(n.out_arrays, n.out_specs):
                specs[aid] = s
        return {
            "fingerprint": region.fingerprint,
            "ops": list(region.ops),
            "copies": region.copies,
            "p": dict(region.p),
            "bound": region.roofline.bound if region.roofline else None,
            "io": {
                "inputs": [[list(specs[a][0]), specs[a][1]] for a in rep.input_ids],
                "outputs": [[list(specs[a][0]), specs[a][1]] for a in rep.output_ids],
            },
            "head_ms": run.head_ms,
            "shipped_ms": run.shipped_ms,
            "roofline_ms": region.roofline.t_roofline_ms if region.roofline else None,
            # the judge edits a named parent, so it must see the lineage sources
            "head": run.head.kernel_id if run.head else None,
            "kernels": {
                spec.kernel_id: _kernel_view(spec)
                for spec in (run.scaffold, run.head, run.shipped)
                if spec is not None
            },
        }

    def _close_rule(self, run: RegionRun, queue: Queue) -> str | None:
        region = run.region
        if run.hypotheses >= self.manifest.budget_per_region:
            return "the region's hypothesis budget is spent"
        if self.total_hypotheses >= self.manifest.budget_total:
            return "the job budget is spent"
        if any(v >= FAIL_STREAK_LIMIT for v in run.fail_streak_by_parent.values()):
            return "5 straight compile or static fails on one parent"
        roof = region.roofline.t_roofline_ms if region.roofline else None
        if roof and run.shipped_ms and run.shipped_ms <= SHIP_ROOFLINE_CLOSE * roof:
            return "shipped is within 5% of the roofline"
        if len(run.recent_ship_gains) >= DIMINISHING_SHIPS and all(
            g < DIMINISHING_PCT for g in run.recent_ship_gains[-DIMINISHING_SHIPS:]
        ):
            return "3 ships in a row under 2% better than the last"
        if run.stale_streak >= STALE_LIMIT:
            return "6 hypotheses in a row without a meaningful win"
        step = min(self.step_ms.values())
        if roof and run.head_ms and (run.head_ms - roof) < 0.01 * step:
            return "head's clock is within 1% of the step of the roofline"
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
            and the patch surface exactly as before this attempt (the
            2026-08-31 audit found all three could be left corrupted)."""
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
                # a scope that already hosts a shipped wrapper accumulates:
                # new wrapper = original module + every splice shipped so far
                prior_install = self.installed.get(scope_path)
                if prior_install:
                    original, prior_splices, prior_kernels = prior_install
                    splices = list(prior_splices) + [splice]
                    kernels = dict(prior_kernels)
                else:
                    original = _resolve(self.model, scope_path)
                    splices = [splice]
                    kernels = {}
                kernels[kernel.kernel_id] = kernel
                emitted = emit_wrapper(trace, scope, splices, f"W_{_safe(scope_path)}")
                cls = _load_class(emitted)
                occupant = swap_install(self.model, scope_path, cls(original, kernels))
                installed_now.append((scope_path, occupant))
                emitted_before.setdefault(scope_path, self.emitted.get(scope_path))
                self.emitted[scope_path] = emitted
                pending_installed[scope_path] = (original, splices, kernels)

            # retrace: the cut must literally be one custom dispatch per copy
            for w in self.manifest.workloads:
                retrace, _ = self.tracer.trace(self.model, self.tensors[w.name])
                spans = [(m.start_seq, m.end_seq) for m in region.members
                         if m.workload == w.name]
                if not spans:
                    continue
                rep_check = verify_retrace(
                    baseline_traces[w.name], retrace, spans,
                    [kernel.kernel_id] * len(spans),
                )
                if not rep_check.ok:
                    raise NotReplayable("; ".join(rep_check.reasons))

            self.tracer.uninstall()
            e2e = run_e2e(
                self.session, self.baseline_model, self.model,
                workloads=[(w.name, self.tensors[w.name]) for w in self.manifest.workloads],
                veto_pairs=8,
            )
            if not e2e.passed:
                self.log.append("e2e_failed", fingerprint=region.fingerprint,
                                checks=[c.__dict__ for c in e2e.checks],
                                veto_passed=e2e.veto_passed)
                unwind()
                return False
            self.installed.update(pending_installed)
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
            if any(covered_by(region, s) for s in shipped_regions):
                self.log.append("region_covered", fingerprint=region.fingerprint)
                continue
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
                close_rule=run.close_rule, hypotheses=run.hypotheses, head_ms=run.head_ms,
            )
            self.log.append("region_closed", fingerprint=region.fingerprint,
                            rule=run.close_rule, shipped=run.shipped is not None,
                            hypotheses=run.hypotheses, head_ms=run.head_ms,
                            shipped_ms=run.shipped_ms,
                            outcomes=self.report.regions[-1]["outcomes"])
            self.candidates.append(_region_line(region, "closed", run))
            self.report.write(self.work_dir / "report.json")  # a crash still leaves the story so far

        # The job's headline number is a ratio, so law 4 applies to it too: the
        # untouched model is re-measured here, interleaved with the patched one,
        # never subtracted from the job-start clock taken on a different machine
        # state. "before" stays in the report as the job-start observation it is.
        for w in self.manifest.workloads:
            tensors = self.tensors[w.name]
            comp = compare(
                self.session,
                lambda t=tensors: self.baseline_model(*t),
                lambda t=tensors: self.model(*t),
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

        self.report.constants = {
            "min_win_ms": MIN_WIN_MS, "budget_per_region": self.manifest.budget_per_region,
            "budget_total": self.manifest.budget_total, "seed": self.manifest.seed,
            "clock_pairs": self.clock_pairs,
            "defaulted": list(self.manifest.defaulted),
        }
        self.report.write(self.work_dir / "report.json")
        return self.report

    def emit_artifact(self, out_dir: str | Path) -> Path:
        from .artifact.emit import emit_artifact

        specs: dict[str, KernelSpec] = {}
        for path in self.emitted:
            wrapper = _resolve(self.model, path)
            specs.update(getattr(wrapper, "_specs", {}))
        return emit_artifact(out_dir, list(specs.values()), list(self.emitted.values()), self.report)


def _safe(path: str) -> str:
    return path.replace(".", "_") or "root"


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


def kernel_from_proposal(contract: RegionContract, kernels: dict, proposal, kernel_id: str) -> KernelSpec:
    """The harness owns names, dtypes, and the call site; the judge supplies
    the body and the launch. A header or template left out means the parent's."""
    parent = kernels.get(proposal.parent_kernel_id)
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
    acceptance envelope (hard law; plan section 10 permits the excess alone)."""
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
