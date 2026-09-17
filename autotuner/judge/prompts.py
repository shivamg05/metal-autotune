"""What the judge sees, rendered from region state (spec "What the judge is
allowed to propose").

The briefing must be plain JSON metadata. Check the entire rendered tree for
numeric acceptance settings so a newly added field cannot bypass the guard.
"""

from __future__ import annotations

import json
import math
from typing import Mapping, Sequence

from autotuner_runtime import grammar

from ..regions.types import Region
from .queue import Queue
from .examples import MOVES
from .schema import ASSOC_TAGS, SUGGESTED_KINDS

# common kinds with one line each; a suggestion list, not a limit
MENU = {
    "on-chip": "keep intermediates in registers or threadgroup memory between stages",
    "specialize": "specialize launch and tiles for this workload's shapes, without breaking other shapes",
    "retile": "change tile sizes, work per thread, SIMD-group layout",
    "re-layout": "re-lay data out, as long as the layout round-trips exactly",
    "algorithm": "same math, different algorithm: split-K, one-pass vs two-pass attention, persistent kernel, loop order",
    "launch": "grid, threadgroup size, threadgroup-memory budget",
    "fix": "repair whatever just failed to compile or match; always legal",
}
assert tuple(MENU) == SUGGESTED_KINDS

LAWS = (
    "boundary dtypes are frozen",
    "hypothesis ids cannot be scaffold, scafix, original, head or shipped; those names belong to the harness",
    "no new quantization or casts the recorded ops do not already contain",
    "no approximate math",
    "shape-specialized kernels fall back outside their supported input signatures",
    "fast-math is pinned by the harness; init_value, math_mode, and streams are not yours to set",
    "preserving edits must match the untouched original bit-for-bit on every tested output and state",
    "changing edits may reorder floating-point evaluation, including fused multiply-add, and must stay within the manifest tolerance against the untouched original",
    "choose target_workload before measurement: whole-model improvement must repeat there, with no resolved slowdown on any other performance workload; gains are not averaged across workloads",
)

# the expression language for grid, threadgroup, output shapes, and fallback
# predicates, verbatim from the one evaluator that runs it
LAUNCH_GRAMMAR_DOC = grammar.__doc__

BODY = (
    "For a single-dispatch proposal, source is the body of one Metal kernel; the harness writes the signature. "
    "Inputs are in0, in1, ... in the order of io.inputs, region outputs are out0, out1, ... "
    "in the order of io.outputs, and scratch buffers you declare are tmp0, tmp1, .... "
    "Naming inN_shape, inN_strides, or inN_ndim in the body makes the harness pass those "
    "buffers. grid is the total thread count per axis, not the threadgroup count. "
    "The kernel must be right at every size of any named dimension, or declare a "
    "fallback_predicate that is true wherever it is not."
)

LEGEND = {
    "target_workload": "optional proposal field: a name from region.timing_workloads. It selects the "
                       "isolated timing case and the whole-model improvement to confirm. If omitted, "
                       "region.default_target_workload is used. Correctness still checks every case.",
    "timing_case": "the captured input case actually timed; additional copies can have different shapes",
    "head_workload": "the workload used for head_ms; compare local timings only within the same workload",
    "shipped_workload": "the workload used for shipped_ms; whole-model results are listed separately for every workload",
    "nonfinite_constants": "In read-only original calls and diagnostics, the strings NaN, Infinity "
                           "and -Infinity represent nonfinite numbers. Executable originals retain "
                           "their numeric values; this is only their JSON display.",
    "p": "share of one model step this region costs, all copies together, per workload",
    "copies": "how many times this op sequence occurs in the model; all copies ship together",
    "bound": "the resource that limits the region: memory (bytes moved), compute (flops), or launch (kernel count)",
    "T_orig_ms": "the library's time for all copies, per workload",
    "T_rep_ms": "the library's time for one copy, per workload; every other ms figure here is one copy",
    "roofline_ms": "an advisory estimate for one copy: a boundary-streaming probe, or an estimated "
                   "arithmetic cost when known and larger; neither is a guaranteed physical minimum",
    "s_max": "estimated speedup opportunity for one copy: T_rep_ms over roofline_ms; a ranking hint",
    "floor_ms": "in a verdict: the boundary-streaming probe clocked beside the kernel; advisory only",
    "head_ms": "one copy's time for head, the kernel being edited; null until it is clocked",
    "shipped_ms": "one copy's time for the installed kernel; null when none is installed",
    "library_ms": "the library's time measured beside a kernel in the same clock",
    "win_ms": "library_ms minus the kernel's time; a region win nominates the kernel. "
              "It ships only after installation, output checks, and a measured whole-model win",
    "sigma_ms": "uncertainty of win_ms",
    "outcome": "failed: a gate rejected the kernel; correct_slower: correct, still a tuning "
               "parent; tentative_ship: region clock won; shipped: the installed model won; "
               "rolled_back: installation or whole-model checks rejected it",
    "model_check": "whole-model result after a region nomination: status, reason, checks and timings. "
                   "not_tested means no whole-model verdict exists. Binding or correctness failures "
                   "cannot replace head; repair the rejected kernel explicitly by its id.",
    "incumbent_screen": "fresh region timing against the kernel already installed for this cut. "
                        "A resolved_regression skips model timing; it is a local screening decision, "
                        "not a measured whole-model regression. The candidate source remains available as a parent.",
    "writing_for": "the queue item your kernel is for; after your mutations it must be the front ready item",
    "budget": "attempts left for this region and for the whole job; a yield is refused while any remain, "
              "and after one free re-ask every reply with nothing to evaluate costs an attempt",
    "history": "every attempt on this region so far, in order, each with its one-line outcome; the "
               "sources you may edit are in kernels",
    "lessons": "what you wrote down for later regions of this job, oldest first; write one (the "
               "optional lesson field of a reply) when a verdict taught something the next region "
               "should know",
    "regions_done": "the regions this job already closed: what shipped there and how many attempts it took",
    "plan_refused": "in a verdict: your last reply was refused for this reason and nothing was evaluated; "
                    "the rest of the verdict is unchanged from the call before",
    "directions": "what the widening round opens from: hardware facts that can pay under this "
                  "region's bound, in no particular order. Each names a structure, what it trades, "
                  "and the bounds it pays under; its kind is the label to use. Add a direction of "
                  "your own when none fits",
    "widening": "the region's first attempts are openers: each written against the scaffold, under "
                "a kind no earlier opener used (a direction's kind, or your own), its hypothesis "
                "stating in one sentence how the work maps onto threads. A repair of a kernel that "
                "failed is allowed meanwhile; any other parent is refused until left reaches 0. "
                "openers: how many the round holds; opened: the kinds already opened",
    "chip": "this machine: gpu_cores, bandwidth_gbps (billions of bytes the whole chip moves per second), "
            "launch_us (one kernel launch), flops_gflops per dtype. A threadgroup runs on one core "
            "and gets one core's share of the bandwidth, so a launch needs threadgroups across every "
            "core, several per core, to move bytes at bandwidth_gbps",
}

# Same keys are removed from gate details by the loop. floor_ms and roofline_ms
# are physical timing estimates and remain visible; floor is a numeric gate.
ENVELOPE_KEYS = frozenset({"rtol", "atol", "tolerance", "tolerances", "kappa", "floor",
                           "changing_floor", "err_library", "err_candidate", "allowance",
                           "cosine_allowance", "floor_max_abs", "floor_cosine"})


def render_region_state(
    *,
    region: Region,
    io_specs: Mapping[str, Mapping[str, Sequence[tuple]]],
    ops: Sequence[Mapping],
    kernels: Mapping[str, Mapping],
    head: str | None,
    shipped: str | None,
    head_ms: float | None,
    shipped_ms: float | None,
    assoc_tag: str,
    queue: Queue,
    last_verdict: Mapping | None,
    writing_for: Mapping | None,
    chip: Mapping | None = None,
    head_floor_ms: float | None = None,
    shipped_floor_ms: float | None = None,
    budget: Mapping | None = None,
    history: Sequence[Mapping] = (),
    lessons: Sequence[Mapping] = (),
    regions_done: Sequence[Mapping] = (),
    default_target_workload: str | None = None,
    head_workload: str | None = None,
    shipped_workload: str | None = None,
    directions: Sequence[Mapping] = (),
    widening: Mapping | None = None,
) -> dict:
    """The one prompt contract, rendered per call. io_specs maps workload ->
    {"inputs": [(shape, dtype)], "outputs": [...]}; ops lists the recorded
    calls with their non-tensor arguments; kernels maps kernel id -> its
    source, launch, and verdict, already reduced to what the judge may see."""
    if assoc_tag not in ASSOC_TAGS:
        raise ValueError(f"assoc_tag {assoc_tag!r} must be one of {list(ASSOC_TAGS)}")
    roofline = region.roofline
    rendered = {
        "region": {
            "fingerprint": region.fingerprint,
            "timing_workloads": sorted(region.workloads),
            "default_target_workload": default_target_workload,
            "ops": list(ops),
            "io": {
                workload: {
                    "inputs": [[list(shape), dtype] for shape, dtype in specs["inputs"]],
                    "outputs": [[list(shape), dtype] for shape, dtype in specs["outputs"]],
                }
                for workload, specs in io_specs.items()
            },
            "copies": region.copies,
            "p": dict(region.p),
            "bound": roofline.bound if roofline else None,
            "T_orig_ms": dict(region.t_orig_ms),
            "T_rep_ms": dict(region.t_rep_ms),
            "roofline_ms": roofline.t_roofline_ms if roofline else None,
            "s_max": roofline.s_max if roofline else None,
            "head_ms": head_ms,
            "shipped_ms": shipped_ms,
            "head_workload": head_workload,
            "shipped_workload": shipped_workload,
            "head_minus_roofline_ms": _distance(head_ms, head_floor_ms, roofline),
            "shipped_minus_roofline_ms": _distance(shipped_ms, shipped_floor_ms, roofline),
        },
        "head": head,
        "shipped": shipped,
        "kernels": {k: dict(v) for k, v in kernels.items()},
        "family": f"assoc-{assoc_tag}",
        "budget": dict(budget or {}),
        "history": [dict(h) for h in history],
        "lessons": [dict(l) for l in lessons],
        "regions_done": [dict(r) for r in regions_done],
        "last_verdict": dict(last_verdict) if last_verdict is not None else None,
        "queue": list(queue.snapshot()),
        "verdicts": queue.verdicts,
        "writing_for": dict(writing_for) if writing_for is not None else None,
        "chip": dict(chip or {}),
        "menu": dict(MENU),
        "moves": list(MOVES),
        "directions": [dict(d) for d in directions],
        "widening": dict(widening or {}),
        "laws": list(LAWS),
        "launch_grammar": LAUNCH_GRAMMAR_DOC,
        "body": ("This is an existing native Metal kernel. For a single-dispatch edit, change its source/header and grid/threadgroup. "
                 "Use the original argument names in native_call.factory; native_call.bindings maps "
                 "each native input to an inN tensor index or a fixed scalar. Launch expressions "
                 "still use inN.shape. Templates, compiler settings, scalars, and output allocations "
                 "are fixed. Omit template and scratch. Preserving edits require bit-identical outputs; "
                 "changing edits are checked against the untouched original using manifest tolerances. "
                 "Every output, including state, is checked. Other input signatures fall back to "
                 "the original. Alternatively, supply the complete ordered-stages format from the response schema; stages use local inN/outN array slots, with explicit intermediate shapes and dtypes. Reproduce the original scalar/template behavior in that code and preserve every region output. Probe headroom is advisory; arbitrary native arithmetic cost is unknown."
                 if region.ops == ("metal_kernel",) else
                 BODY + " When reference_sequence is present, it is the untouched multi-call starter, "
                 "not one editable Metal body. Its nodes expose every original operation, argument, "
                 "captured custom source/header and launch, and tensor connections by array id. "
                 "Write a replacement body using the region inN/outN ABI. Do not concatenate original "
                 "bodies blindly: separate launches may supply synchronization a single dispatch lacks. "
                 "A proposal may instead supply explicit ordered stages, following the stage schema. "
                 "All region outputs and state must survive. A fixed input_signature supplies automatic "
                 "fallback outside the captured shapes and dtypes. reference_sequence and input_signature "
                 "are harness-owned, never proposal fields."),
        "legend": dict(LEGEND),
    }
    validate_metadata(rendered)
    return rendered


def diagnostic_metadata(value):
    """Encode diagnostic nonfinite values as explicit text, never JSON numbers.

    Apply to gate details after hiding acceptance settings, and to read-only
    original-call metadata. Executable specs are never modified. Measurements
    and executable launch expressions still pass the strict validator unchanged.
    """
    if isinstance(value, Mapping):
        return {key: diagnostic_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_metadata(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def validate_metadata(payload: Mapping) -> None:
    """Check the complete payload before it reaches any judge transport."""
    _refuse_tolerances(payload, "payload")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise TypeError(f"the judge prompt must be plain JSON metadata: {e}") from e


def _distance(clock_ms: float | None, floor_ms: float | None, roofline) -> float | None:
    """How far a kernel sits above its floor: the probe clocked beside it
    when there is one, else the region's priced roofline."""
    if clock_ms is None:
        return None
    if floor_ms:
        return clock_ms - floor_ms
    return clock_ms - roofline.t_roofline_ms if roofline is not None else None


def _refuse_tolerances(obj: object, where: str) -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if isinstance(key, str) and key.lower() in ENVELOPE_KEYS:
                raise ValueError(f"tolerance values never reach the judge: {where} carries {key!r}")
            _refuse_tolerances(value, f"{where}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _refuse_tolerances(value, f"{where}[{i}]")
