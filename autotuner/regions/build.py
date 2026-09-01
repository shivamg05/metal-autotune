"""Region building (spec "Regions"): singletons plus grown chains, view
absorption, slice-write termination, the rejection rules, per-stretch liveness.

Candidates come from the recorded ops alone; the module tree only supplies
delivery scopes. Chains are maximal runs of consecutive calls between
barriers, kept with their prefixes; not every subset is enumerated.
"""

from __future__ import annotations

from ..trace.recorder import ArrayRef, OPAQUE_OP
from ..trace.types import Retention, Trace, TraceNode
from .types import Stretch

# Ops that only reinterpret layout: absorbed into regions, never regions alone.
VIEW_OPS = frozenset({
    "mx.reshape", "mx.transpose", "mx.squeeze", "mx.expand_dims",
    "mx.broadcast_to", "mx.flatten", "mx.swapaxes", "mx.moveaxis",
    "mx.atleast_1d", "mx.atleast_2d", "mx.atleast_3d", "mx.split",
    "mx.stop_gradient", "mx.as_strided",
    "array.reshape", "array.transpose", "array.squeeze", "array.flatten",
    "array.swapaxes", "array.moveaxis", "array.split", "array.T",
})


def _contains_array_ref(obj: object) -> bool:
    if isinstance(obj, ArrayRef):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_contains_array_ref(v) for v in obj)
    if isinstance(obj, dict):
        return any(_contains_array_ref(v) for v in obj.values())
    if isinstance(obj, slice):
        return any(_contains_array_ref(v) for v in (obj.start, obj.stop, obj.step))
    return False


def is_view(node: TraceNode) -> bool:
    if node.op in VIEW_OPS:
        return True
    if node.op == "array.__getitem__":
        # a basic slice read is a view; a key containing arrays is a gather
        key_args = node.scalar_args["args"][1:]
        return not any(_contains_array_ref(a) for a in key_args)
    return False


def _is_barrier(node: TraceNode, trace: Trace) -> bool:
    """A node no stretch may contain: a slice write (edits memory something
    else may hold), an opaque compiled call, or the producer of a value the
    model's own python kept (the wrapper can never reach that reference)."""
    if node.op == "array.__setitem__":
        return True
    if node.op == OPAQUE_OP:
        return True
    for out in node.out_arrays:
        if trace.liveness[out].kind is Retention.PYTHON_RETAINED:
            return True
    return False


def _boundary(trace: Trace, workload: str, start: int, end: int) -> Stretch:
    """Derive one stretch's inputs, live outputs, and scope."""
    nodes = trace.nodes[start:end + 1]
    produced: set[int] = set()
    inputs: list[int] = []
    for node in nodes:
        for aid in node.in_arrays:
            if aid not in produced and aid not in inputs:
                inputs.append(aid)
        produced.update(node.out_arrays)

    outputs: list[int] = []
    step_outs = set(trace.step_outputs)
    for node in nodes:
        for aid in node.out_arrays:
            live = trace.liveness[aid]
            consumed_outside = any(s > end for s in live.consumed_by)
            if aid in step_outs or consumed_outside:
                if aid not in outputs:
                    outputs.append(aid)

    stacks = [n.module_stack for n in nodes]
    common: list[str] = []
    for parts in zip(*stacks):
        if all(p == parts[0] for p in parts):
            common.append(parts[0])
        else:
            break
    return Stretch(
        workload=workload,
        start_seq=start,
        end_seq=end,
        input_ids=tuple(inputs),
        output_ids=tuple(outputs),
        scope_stack=tuple(common) or (trace.nodes[0].module_stack[0],),
    )


# A stretch longer than this many compute ops is implausible as one kernel;
# the cap keeps candidate counts linear-ish on real models. Tunable, recorded.
MAX_STRETCH_COMPUTE_OPS = 64


def build_stretches(trace: Trace, workload: str) -> list[Stretch]:
    """Every candidate stretch in one workload's trace: singletons for each
    non-view, non-barrier node, plus chains grown from anchors. Anchors are
    the positions after a barrier and every module-address change, so the
    per-layer sequence the copy-grouping rule prices together is always a
    candidate, without enumerating every subset. Chains grow to the next
    barrier (or the cap), keeping each compute-ending prefix."""
    n = len(trace.nodes)
    barrier = [_is_barrier(node, trace) for node in trace.nodes]

    anchors = set()
    for i, node in enumerate(trace.nodes):
        if barrier[i]:
            continue
        if i == 0 or barrier[i - 1] or node.module_address != trace.nodes[i - 1].module_address:
            anchors.add(i)

    stretches: list[Stretch] = []
    seen_spans: set[tuple[int, int]] = set()

    def add(start: int, end: int) -> None:
        span = (start, end)
        if span not in seen_spans:
            seen_spans.add(span)
            stretches.append(_boundary(trace, workload, start, end))

    for i, node in enumerate(trace.nodes):
        if not barrier[i] and not is_view(node):
            add(i, i)

    for a in sorted(anchors):
        compute_seen = 0
        for j in range(a, n):
            if barrier[j]:
                break
            if not is_view(trace.nodes[j]):
                compute_seen += 1
                if compute_seen > MAX_STRETCH_COMPUTE_OPS:
                    break
                if j > a:
                    add(a, j)

    return stretches
