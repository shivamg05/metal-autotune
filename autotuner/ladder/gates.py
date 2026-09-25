"""The ladder, orchestrated.

Gate 1 (static checks) runs in-parent with no subprocess. Gates 2-8 run in ONE
validate-mode child (shader validation on, stderr parsed for invalid access
lines). Gate 9 runs in ONE score-mode child that first re-runs smoke and
determinism, because validation recompiles pipelines and the timed pipeline
must re-prove itself in the clean process. First failure stops.

Verdict mapping: failed (any gate), correct_slower (all correctness gates
pass but the clock shows no ship-margin win, or the clock was not run), or
tentative_ship (median win beats max(1% of the freshly re-measured library
region time, 3 sigma of the interleaved samples) and the absolute floor).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from autotuner.ladder.static_checks import RegionContract, check
from autotuner.measure.session import Session
from autotuner.sandbox.protocol import EvalSetSpec, LadderSpec, require_responsive, run_job
from autotuner_runtime.kernels import KernelSpec
from autotuner_runtime.numeric import validate_tolerances

@dataclass
class EvalSet:
    """One eval set: k saved input files and k library-reference files
    (safetensors keyed "a<array_id>"). The first set is the nominated timing case and
    must carry t_library_ms; correctness_only sets are sweep instances, and
    nodes_json carries the span at that size when the shapes differ."""

    label: str
    inputs_paths: list[str]
    reference_paths: list[str]
    t_library_ms: float | None = None
    correctness_only: bool = False
    nodes_json: str | None = None


@dataclass
class LadderJob:
    """Harness-held numeric policy: exact preserving outputs; manifest tolerance
    for changed floating evaluation. None selects defaults per output dtype.
    kappa/changing_floor are legacy wire fields and no longer affect acceptance.
    """

    kernel: KernelSpec
    contract: RegionContract
    assoc_tag: str                    # "preserving" | "changing"
    nodes_json: str                   # the primary region span (serialize.nodes_to_json)
    input_ids: tuple[int, ...]        # kernel input order = these ids' order
    output_ids: tuple[int, ...]
    eval_sets: list[EvalSet]
    tolerances: tuple[float, float] | None  # None: per-output dtype defaults
    kappa: float = 1.25
    changing_floor: float | None = None
    min_win_ms: float = 0.03
    run_clock: bool = True            # False for scaffold runs (gates 1-8 only)
    clock_pairs: int = 32             # ABBA pairs behind the ship clock
    baseline: str = "plain"  # what the library arm runs as: plain ops or one compiled graph
    timeout_s: float = 300.0
    seed: int = 0
    weight_inputs: tuple[bool, ...] = ()  # per input id: a model weight, never perturbed
    compute_floor_ms: float = 0.0     # the roofline's flops term; the child's probe covers bytes and launch
    timing_incumbent: KernelSpec | None = None  # accepted kernel for this exact cut; never a correctness reference


@dataclass(frozen=True)
class LadderResult:
    outcome: str                      # "failed" | "correct_slower" | "tentative_ship"
    failed_gate: str | None
    detail: dict
    region_ms: float | None
    library_ms: float | None
    win_ms: float | None
    sigma_ms: float | None
    gates_passed: list[str] = field(default_factory=list)
    floor_ms: float | None = None     # the floor probe clocked beside the library in the same child


def run_ladder(job: LadderJob, *, session: Session | None = None) -> LadderResult:
    """An owning session permits worker cooling to overlap parent CPU work.

    Standalone callers retain the blocking behavior: no GPU debt escapes.
    """
    _validate(job)

    failures = check(job.kernel, job.contract)
    if failures:
        return LadderResult(
            "failed", "static",
            {"failures": [{"check": f.check, "detail": f.detail} for f in failures]},
            None, None, None, None, [],
        )
    gates = ["static"]

    def phase(mode):
        spec = _spec(job, mode)
        if session is not None:
            spec = replace(spec, defer_cooling=True)
            session.wait_ready()
        verdict = run_job(spec, mode, job.timeout_s)
        # Accept the deadline even for an ordinary failed correctness gate.
        # Recovered timeouts supply a conservative cooling handoff too.
        if session is not None and verdict.failed_gate != "subprocess":
            if "cooling_ready_at" not in verdict.timing:
                raise RuntimeError("worker returned without its required cooling deadline")
            session.adopt_cooling(verdict.timing.pop("cooling_ready_at"))
        return verdict

    v = phase("validate")
    require_responsive(v)
    if not v.passed:
        return LadderResult("failed", v.failed_gate, dict(v.detail),
                            None, None, None, None, gates + list(v.gates_passed))
    if "validation" in v.detail:
        # spec gate 3: an out-of-bounds access that still produced plausible
        # values; the shader-validation stderr evidence is the verdict
        return LadderResult("failed", "poison", {"validation": v.detail["validation"]},
                            None, None, None, None, gates + ["compile"])
    gates += list(v.gates_passed)
    detail = dict(v.detail)
    detail["pacing"] = {"validate": {k: v.timing[k] for k in ("pacing_work_s", "pacing_idle_s") if k in v.timing}}

    if not job.run_clock:
        return LadderResult("correct_slower", None, detail, None, None, None, None, gates)

    s = phase("score")
    require_responsive(s)
    if not s.passed:
        d = dict(s.detail)
        d["phase"] = "score"
        return LadderResult("failed", s.failed_gate, d, None, None, None, None, gates)
    gates.append("clock")
    detail.update(s.detail)
    detail["pacing"]["score"] = {k: s.timing[k] for k in ("pacing_work_s", "pacing_idle_s") if k in s.timing}
    t = s.timing
    outcome = "tentative_ship" if s.detail.get("ship") else "correct_slower"
    return LadderResult(outcome, None, detail, t.get("region_ms"), t.get("library_ms"),
                        t.get("win_ms"), t.get("sigma_ms"), gates, floor_ms=t.get("floor_ms"))


def _validate(job: LadderJob) -> None:
    if job.assoc_tag not in ("preserving", "changing"):
        raise ValueError(f"assoc_tag must be 'preserving' or 'changing', got {job.assoc_tag!r}")
    if not job.eval_sets:
        raise ValueError("a ladder job needs at least one eval set")
    if not job.output_ids:
        raise ValueError("the region has no live outputs: nothing it computes is ever used")
    first = job.eval_sets[0]
    if first.correctness_only:
        raise ValueError("the first eval set must be a performance workload, not a sweep instance")
    if first.t_library_ms is None:
        raise ValueError("the first eval set must carry t_library_ms")
    if len(first.inputs_paths) != len(first.reference_paths):
        raise ValueError("inputs_paths and reference_paths must pair up")
    if job.tolerances is not None:
        validate_tolerances(job.tolerances)


def _spec(job: LadderJob, phase: str) -> LadderSpec:
    return LadderSpec(
        kernel=json.loads(job.kernel.to_json()),
        assoc_tag=job.assoc_tag,
        nodes_json=job.nodes_json,
        input_ids=tuple(job.input_ids),
        output_ids=tuple(job.output_ids),
        eval_sets=tuple(
            EvalSetSpec(
                label=e.label,
                inputs_paths=tuple(e.inputs_paths),
                reference_paths=tuple(e.reference_paths),
                t_library_ms=e.t_library_ms,
                correctness_only=e.correctness_only,
                nodes_json=e.nodes_json,
            )
            for e in job.eval_sets
        ),
        tolerances=({"rtol": job.tolerances[0], "atol": job.tolerances[1]}
                    if job.tolerances is not None else {}),
        kappa=job.kappa,
        changing_floor=job.changing_floor,
        min_win_ms=job.min_win_ms,
        phase=phase,
        baseline=job.baseline,
        clock_pairs=job.clock_pairs,
        seed=job.seed,
        weight_inputs=tuple(job.weight_inputs),
        compute_floor_ms=job.compute_floor_ms,
        timing_incumbent=json.loads(job.timing_incumbent.to_json()) if job.timing_incumbent else None,
    )
