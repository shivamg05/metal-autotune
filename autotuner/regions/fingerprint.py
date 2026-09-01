"""Region fingerprints: copy grouping by canonical form.

Two stretches are copies of one region iff their canonical forms match: op
sequence, dtypes, ranks, non-shape scalar args, and internal edge structure,
with array identities replaced by role indices. Activation shapes and
shape-derived scalar args (reshape targets, slice bounds, split sizes) are
deliberately not part of the form, so the same sequence at two sizes is one
region. Weights count by role, dtype, and shape: a weight never moves with a
named dimension, and a projection against a different weight shape is a
different kernel to write, price, and check.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

from ..trace.recorder import ArrayRef
from ..trace.types import Trace
from .build import VIEW_OPS
from .types import Region, Stretch

# Ops whose non-array scalar args are shape-derived and therefore excluded
# from the canonical form.
_SHAPE_ARG_OPS = VIEW_OPS | {"array.__getitem__"}


def _canon_scalar(obj: object) -> object:
    if isinstance(obj, ArrayRef):
        return ("ref", obj.index)
    if isinstance(obj, (list, tuple)):
        return tuple(_canon_scalar(v) for v in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _canon_scalar(v)) for k, v in obj.items()))
    if isinstance(obj, slice):
        return ("slice", _canon_scalar(obj.start), _canon_scalar(obj.stop), _canon_scalar(obj.step))
    return repr(obj)


def canonical_form(trace: Trace, stretch: Stretch) -> tuple:
    nodes = trace.nodes[stretch.start_seq:stretch.end_seq + 1]
    roles: dict[int, tuple] = {}
    x_count = w_count = 0
    for pos, node in enumerate(nodes):
        for i, aid in enumerate(node.out_arrays):
            roles[aid] = ("i", pos, i)

    form = []
    for pos, node in enumerate(nodes):
        in_roles = []
        for aid, (shape, _) in zip(node.in_arrays, node.in_specs):
            if aid not in roles:
                if aid in trace.weights:
                    roles[aid] = ("w", w_count, tuple(shape))
                    w_count += 1
                else:
                    roles[aid] = ("x", x_count)
                    x_count += 1
            in_roles.append(roles[aid])
        if node.op in _SHAPE_ARG_OPS:
            scalars = None
        else:
            scalars = (
                tuple(_canon_scalar(a) for a in node.scalar_args["args"]),
                _canon_scalar(node.scalar_args["kwargs"]),
            )
        form.append((
            node.op,
            tuple(in_roles),
            tuple((len(s), d) for s, d in node.in_specs),   # rank + dtype, never shape
            tuple((len(s), d) for s, d in node.out_specs),
            scalars,
        ))
    return tuple(form)


def fingerprint(trace: Trace, stretch: Stretch) -> str:
    return hashlib.sha256(repr(canonical_form(trace, stretch)).encode()).hexdigest()[:16]


def group_copies(traces: dict[str, Trace], stretches: dict[str, list[Stretch]]) -> list[Region]:
    """All copies of the same op sequence across the model and across
    workloads are one region, with one hypothesis history and one wrapper."""
    by_print: dict[str, Region] = {}
    taken: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for workload, trace in traces.items():
        for stretch in stretches[workload]:
            fp = fingerprint(trace, stretch)
            spans = taken[(fp, workload)]
            if any(not (stretch.end_seq < a or stretch.start_seq > b) for a, b in spans):
                continue  # a periodic op sequence: copies must not overlap
            spans.append((stretch.start_seq, stretch.end_seq))
            if fp not in by_print:
                ops = tuple(
                    n.op for n in trace.nodes[stretch.start_seq:stretch.end_seq + 1]
                )
                by_print[fp] = Region(fingerprint=fp, ops=ops)
            by_print[fp].members.append(stretch)
    return list(by_print.values())
