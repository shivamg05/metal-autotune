"""Trace data model. The recorder fills these; freeze checks them.

array_id is a stable identity the recorder assigns while it holds every array
alive for the pass. (module_address, position_in_module) is the install address
for a wrapper; position counts recorded calls per op name within the address so
the pair survives partial patch surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

Spec = tuple[tuple[int, ...], str]  # (shape, dtype name)
STATE_PREFIX = "state:"             # a call on an object holding model state, recorded whole


@dataclass(frozen=True)
class TraceNode:
    seq: int
    op: str
    in_arrays: tuple[int, ...]
    out_arrays: tuple[int, ...]
    in_specs: tuple[Spec, ...]
    out_specs: tuple[Spec, ...]
    scalar_args: Mapping[str, object]
    module_address: str
    position_in_module: int
    # full enclosing call chain, outermost first; module_address == stack[-1].
    # Region delivery scopes are common prefixes of member stacks.
    module_stack: tuple[str, ...] = ()


class Retention(Enum):
    CONSUMED = "consumed"          # read by later recorded calls only
    STEP_OUTPUT = "step_output"    # in the returned tree (may also be consumed)
    PYTHON_RETAINED = "python_retained"  # the model itself kept a reference


@dataclass(frozen=True)
class Liveness:
    kind: Retention
    consumed_by: tuple[int, ...]  # seqs of recorded consumers, always populated


@dataclass(frozen=True)
class Trace:
    nodes: tuple[TraceNode, ...]
    edges: Mapping[int, tuple[int, ...]]        # producer seq -> consumer seqs
    step_outputs: tuple[int, ...]               # array_ids the step returned
    weights: frozenset[int]
    inputs: frozenset[int]
    liveness: Mapping[int, Liveness]            # array_id -> liveness
    weight_paths: Mapping[int, str] = field(default_factory=dict)  # weight id -> model path
    in_pass_evaluation: bool = False            # model evaluated mid-record (memory warning)
    eval_sites: tuple[tuple[str, ...], ...] = ()  # addr stacks where the model evaluated
    scope_calls: tuple["ScopeCall", ...] = ()   # per module call: entry/exit record

    def python_retained(self) -> list[int]:
        """Arrays the model itself kept after the pass: a value it stored on
        itself, a buffer it wrote in place outside any state call."""
        return sorted(a for a, live in self.liveness.items()
                      if live.kind is Retention.PYTHON_RETAINED)

    def state_calls(self) -> list[int]:
        """Seqs of the calls that changed an object's state (a KV cache
        write), recorded whole because the wrapper replays them by calling
        the same method on the same object."""
        return [n.seq for n in self.nodes if n.op.startswith(STATE_PREFIX)]

    def span_specs(self, start: int, end: int) -> dict[int, Spec]:
        """array id -> (shape, dtype) for every array the span's calls touch."""
        specs: dict[int, Spec] = {}
        for node in self.nodes[start:end + 1]:
            for aid, spec in zip(node.in_arrays, node.in_specs):
                specs.setdefault(aid, spec)
            for aid, spec in zip(node.out_arrays, node.out_specs):
                specs[aid] = spec
        return specs


@dataclass(frozen=True)
class ScopeCall:
    """One recorded module (or top-level) call: what entered and what left.
    The wrapper generator emits __call__ from these templates."""

    address: str                        # e.g. "layers.3@0"; "" plus index for the root
    stack: tuple[str, ...]
    args_template: tuple                # ArrayRefs index into arg_ids, ObjectRefs into obj_ids
    kwargs_template: Mapping[str, object]
    arg_ids: tuple[int, ...]
    out_template: object                # ArrayRefs index into out_ids
    out_ids: tuple[int, ...]
    obj_ids: tuple[int, ...] = ()       # id() of each non-array argument object (a cache)


class TraceIncomplete(RuntimeError):
    """An array appeared from nowhere or a step output has no producer: some
    MLX entry point was not wrapped. Names the offending call."""
