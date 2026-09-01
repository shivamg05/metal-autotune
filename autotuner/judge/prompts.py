"""What the judge sees, rendered from region state (plan section 10).

Secrecy is structural: this renderer's signature has no parameter that could
carry a tensor or a tolerance value, every argument is plain metadata, and
the rendered dict must round-trip through JSON. As a backstop it refuses
verdict detail containing tolerance-named keys, loudly, so a loop bug cannot
leak them quietly.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from autotuner_runtime import grammar

from ..regions.types import Region
from .queue import FamilyBook, Queue
from .schema import ASSOC_TAGS, KINDS

# the spec's menu, one line per kind
MENU = {
    "on-chip": "keep intermediates in registers or threadgroup memory between stages",
    "specialize": "specialize launch and tiles for this workload's shapes, without breaking other shapes",
    "retile": "change tile sizes, work per thread, SIMD-group layout",
    "re-layout": "re-lay data out, as long as the layout round-trips exactly",
    "algorithm": "same math, different algorithm: split-K, one-pass vs two-pass attention, persistent kernel, loop order",
    "launch": "grid, threadgroup size, threadgroup-memory budget",
    "fix": "repair whatever just failed to compile or match; always legal",
}
assert tuple(MENU) == KINDS

LAWS = (
    "boundary dtypes are frozen",
    "no new quantization or casts the recorded ops do not already contain",
    "no approximate math",
    "shape-specialized kernels declare a fallback predicate",
    "fast-math is pinned by the harness; init_value, math_mode, and streams are not yours to set",
)

# the expression language for grid/threadgroup/fallback_predicate, verbatim
# from the one evaluator that runs it
LAUNCH_GRAMMAR_DOC = grammar.__doc__

_TOLERANCE_KEYS = frozenset({"rtol", "atol", "tolerance", "tolerances"})


def render_region_state(
    *,
    region: Region,
    io_specs: Mapping[str, Mapping[str, Sequence[tuple]]],
    assoc_tag: str,
    families: FamilyBook,
    queue: Queue,
    head_ms: float | None = None,
    shipped_ms: float | None = None,
    parent: Mapping | None = None,
    last_verdict: Mapping | None = None,
) -> dict:
    """The JSON prompt contract, rendered per call. io_specs maps workload ->
    {"inputs": [(shape, dtype)], "outputs": [...]}; parent is the kernel being
    edited (source + launch expressions + its verdict) and last_verdict the
    most recent outcome with gate detail, both already plain JSON data."""
    if assoc_tag not in ASSOC_TAGS:
        raise ValueError(f"assoc_tag {assoc_tag!r} must be one of {list(ASSOC_TAGS)}")
    _refuse_tolerances(parent, "parent")
    _refuse_tolerances(last_verdict, "last_verdict")

    roofline = region.roofline
    rendered = {
        "region": {
            "io": {
                workload: {
                    "inputs": [[list(shape), dtype] for shape, dtype in specs["inputs"]],
                    "outputs": [[list(shape), dtype] for shape, dtype in specs["outputs"]],
                }
                for workload, specs in io_specs.items()
            },
            "p": dict(region.p),
            "copies": region.copies,
            "bound": roofline.bound if roofline else None,
            "T_orig_ms": dict(region.t_orig_ms),
            "s_max": roofline.s_max if roofline else None,
            "roofline_ms": roofline.t_roofline_ms if roofline else None,
            "head_minus_roofline_ms": _distance(head_ms, roofline),
            "shipped_minus_roofline_ms": _distance(shipped_ms, roofline),
        },
        "family": f"assoc-{assoc_tag}",
        "families": {"per_family": families.state(), "climbing": families.climbing},
        "parent": dict(parent) if parent is not None else None,
        "last_verdict": dict(last_verdict) if last_verdict is not None else None,
        "queue": list(queue.snapshot()),
        "menu": dict(MENU),
        "launch_grammar": LAUNCH_GRAMMAR_DOC,
        "laws": list(LAWS),
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
