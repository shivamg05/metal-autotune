"""Freeze: turn the recorder's raw call list into a checked Trace.

Derives edges and liveness, then runs the completeness check from the spec:
every array feeding a recorded call is a workload input, a model weight, or an
earlier recorded call's output, and every step output comes from a recorded
call. Anything else means an unwrapped entry point, and the freeze aborts
naming the call that received it.
"""

from __future__ import annotations

from collections import defaultdict
from types import MappingProxyType
from typing import Iterable, Sequence

from .optable import MUTATING_METHODS
from .types import Liveness, Retention, Trace, TraceIncomplete, TraceNode


def freeze(
    nodes: Sequence[TraceNode],
    inputs: Iterable[int],
    weights: Iterable[int],
    step_outputs: Sequence[int],
    retained: Iterable[int] = (),
    weight_paths: dict[int, str] | None = None,
    in_pass_evaluation: bool = False,
    eval_sites: tuple = (),
    scope_calls: tuple = (),
    evaluated: Iterable[int] = (),
) -> Trace:
    inputs = frozenset(inputs)
    weights = frozenset(weights)
    retained = frozenset(retained)
    evaluated = frozenset(evaluated)

    produced_by: dict[int, int] = {}  # array_id -> producer seq
    for node in nodes:
        for out in node.out_arrays:
            produced_by[out] = node.seq

    consumers: dict[int, list[int]] = defaultdict(list)  # array_id -> consumer seqs
    for node in nodes:
        for arr in node.in_arrays:
            known = arr in inputs or arr in weights
            if not known:
                producer = produced_by.get(arr)
                if producer is None or producer >= node.seq:
                    raise TraceIncomplete(
                        f"array {arr} fed {node.op!r} (seq {node.seq}, at "
                        f"{node.module_address or '<top>'}) but is not a workload input, "
                        f"a weight, or an earlier recorded call's output; some MLX entry "
                        f"point is unwrapped (an mx.array(...) constant records only when "
                        f"small, finite and met before any in-pass evaluation)"
                    )
            consumers[arr].append(node.seq)

    for arr in step_outputs:
        if arr not in produced_by and arr not in inputs and arr not in weights:
            raise TraceIncomplete(
                f"step output array {arr} was not produced by any recorded call"
            )

    edges: dict[int, list[int]] = defaultdict(list)  # producer seq -> consumer seqs
    for arr, producer in produced_by.items():
        for consumer in consumers.get(arr, ()):
            if consumer > producer:
                edges[producer].append(consumer)

    step_output_set = frozenset(step_outputs)
    liveness: dict[int, Liveness] = {}
    for arr in produced_by:
        consumed = tuple(sorted(consumers.get(arr, ())))
        # Retained wins over everything: a kept reference is unswappable no
        # matter who else reads the value. Step-output-ness stays visible via
        # trace.step_outputs.
        if arr in retained:
            kind = Retention.PYTHON_RETAINED
        elif arr in step_output_set:
            kind = Retention.STEP_OUTPUT
        else:
            kind = Retention.CONSUMED
        liveness[arr] = Liveness(kind=kind, consumed_by=consumed)

    # Backward from what the step needs: its outputs, what the model kept or
    # evaluated, and every call with an effect (a state call, an in-place
    # write). A call none of those reach never runs under lazy evaluation.
    from .recorder import STATE_PREFIX  # the recorder imports this module
    needed = set(step_output_set) | retained | evaluated
    dead = []
    for node in reversed(nodes):
        effect = node.op.startswith(STATE_PREFIX) or node.op.removeprefix("array.") in MUTATING_METHODS
        if effect or any(out in needed for out in node.out_arrays):
            needed.update(node.in_arrays)
        else:
            dead.append(node.seq)

    return Trace(
        nodes=tuple(nodes),
        edges=MappingProxyType({k: tuple(sorted(set(v))) for k, v in edges.items()}),
        step_outputs=tuple(step_outputs),
        weights=weights,
        inputs=inputs,
        liveness=MappingProxyType(liveness),
        weight_paths=MappingProxyType(weight_paths or {}),
        in_pass_evaluation=in_pass_evaluation,
        eval_sites=tuple(eval_sites),
        scope_calls=tuple(scope_calls),
        evaluated=evaluated,
        dead=frozenset(dead),
    )
