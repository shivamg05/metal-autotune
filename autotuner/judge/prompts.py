"""What the judge sees, rendered from region state (spec "What the judge is
allowed to propose").

Secrecy is structural: every argument is plain metadata, nothing here can
carry a tensor or a tolerance, and the rendered dict must round-trip through
JSON. As a backstop the renderer refuses verdict detail that names a
tolerance, loudly, so a loop bug cannot leak one quietly.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from autotuner_runtime import grammar

from ..regions.types import Region
from .queue import FamilyBook, Queue
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
    "no new quantization or casts the recorded ops do not already contain",
    "no approximate math",
    "shape-specialized kernels declare a fallback predicate",
    "fast-math is pinned by the harness; init_value, math_mode, and streams are not yours to set",
    "a kernel is correct only if every output matches the library at every input the checks try",
)

# the expression language for grid, threadgroup, output shapes, and fallback
# predicates, verbatim from the one evaluator that runs it
LAUNCH_GRAMMAR_DOC = grammar.__doc__

BODY = (
    "source is the body of one Metal kernel; the harness writes the signature. "
    "Inputs are in0, in1, ... in the order of io.inputs, region outputs are out0, out1, ... "
    "in the order of io.outputs, and scratch buffers you declare are tmp0, tmp1, .... "
    "Naming inN_shape, inN_strides, or inN_ndim in the body makes the harness pass those "
    "buffers. grid is the total thread count per axis, not the threadgroup count. "
    "The kernel must be right at every size of any named dimension, or declare a "
    "fallback_predicate that is true wherever it is not."
)

LEGEND = {
    "p": "share of one model step this region costs, all copies together, per workload",
    "copies": "how many times this op sequence occurs in the model; all copies ship together",
    "bound": "the resource that limits the region: memory (bytes moved), compute (flops), or launch (kernel count)",
    "T_orig_ms": "the library's time for all copies, per workload",
    "T_rep_ms": "the library's time for one copy, per workload; every other ms figure here is one copy",
    "roofline_ms": "the physical limit for one copy on this chip",
    "s_max": "speedup ceiling for one copy: T_rep_ms over roofline_ms",
    "head_ms": "one copy's time for head, the kernel being edited; null until it is clocked",
    "shipped_ms": "one copy's time for the installed kernel; null when none is installed",
    "library_ms": "the library's time measured beside a kernel in the same clock",
    "win_ms": "library_ms minus the kernel's time; a ship needs a win past the margin",
    "sigma_ms": "uncertainty of win_ms",
    "writing_for": "the queue item your kernel is for; after your mutations it must be the front ready item",
    "chip": "this machine: gpu_cores, bandwidth_gbps (bytes the whole chip moves per second), "
            "launch_us (one kernel launch), flops_gflops per dtype. A threadgroup runs on one core "
            "and gets one core's share of the bandwidth, so a launch needs threadgroups across every "
            "core, several per core, to move bytes at bandwidth_gbps",
}

_TOLERANCE_KEYS = frozenset({"rtol", "atol", "tolerance", "tolerances"})


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
    families: FamilyBook,
    queue: Queue,
    last_verdict: Mapping | None,
    writing_for: Mapping | None,
    chip: Mapping | None = None,
) -> dict:
    """The one prompt contract, rendered per call. io_specs maps workload ->
    {"inputs": [(shape, dtype)], "outputs": [...]}; ops lists the recorded
    calls with their non-tensor arguments; kernels maps kernel id -> its
    source, launch, and verdict, already reduced to what the judge may see."""
    if assoc_tag not in ASSOC_TAGS:
        raise ValueError(f"assoc_tag {assoc_tag!r} must be one of {list(ASSOC_TAGS)}")
    _refuse_tolerances(kernels, "kernels")
    _refuse_tolerances(last_verdict, "last_verdict")

    roofline = region.roofline
    rendered = {
        "region": {
            "fingerprint": region.fingerprint,
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
            "head_minus_roofline_ms": _distance(head_ms, roofline),
            "shipped_minus_roofline_ms": _distance(shipped_ms, roofline),
        },
        "head": head,
        "shipped": shipped,
        "kernels": {k: dict(v) for k, v in kernels.items()},
        "family": f"assoc-{assoc_tag}",
        "families": {"per_family": families.state(), "climbing": families.climbing},
        "last_verdict": dict(last_verdict) if last_verdict is not None else None,
        "queue": list(queue.snapshot()),
        "verdicts": queue.verdicts,
        "writing_for": dict(writing_for) if writing_for is not None else None,
        "chip": dict(chip or {}),
        "menu": dict(MENU),
        "moves": list(MOVES),
        "laws": list(LAWS),
        "launch_grammar": LAUNCH_GRAMMAR_DOC,
        "body": BODY,
        "legend": dict(LEGEND),
    }
    try:
        json.dumps(rendered)
    except TypeError as e:
        raise TypeError(f"the judge prompt must be plain JSON metadata: {e}")
    return rendered


def _distance(clock_ms: float | None, roofline) -> float | None:
    if clock_ms is None or roofline is None:
        return None
    return clock_ms - roofline.t_roofline_ms


def _refuse_tolerances(obj: object, where: str) -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if isinstance(key, str) and key.lower() in _TOLERANCE_KEYS:
                raise ValueError(f"tolerance values never reach the judge: {where} carries {key!r}")
            _refuse_tolerances(value, f"{where}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _refuse_tolerances(value, f"{where}[{i}]")
