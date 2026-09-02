"""The static replayability screen and identity certification (plan 5.2).

The screen is content-monotone: anything disqualifying inside the smallest
enclosing scope is inside every ancestor too, so one check at the region's own
scope decides "no certified delivery scope".

Identity certification is the measured fact that a scope's replay is
invisible: the identity wrapper (same replay, no kernel change) must produce
bitwise-identical outputs everywhere the scope fires, including across
repeated calls of a stateful step, and its retrace must match the baseline
under the span-map projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import mlx.core as mx

from ..trace.recorder import OPAQUE_OP
from ..trace.types import Retention, ScopeCall, Trace
from ..trace.walk import flatten_arrays
from .emit import NotReplayable, emit_wrapper, scope_nodes


def find_scope_call(trace: Trace, stack: tuple[str, ...]) -> ScopeCall | None:
    for sc in trace.scope_calls:
        if sc.stack == stack:
            return sc
    return None


def screen_scope(trace: Trace, stack: tuple[str, ...]) -> str | None:
    """None if the scope qualifies, else the named reason it cannot host a
    generated replay wrapper."""
    scope = find_scope_call(trace, stack)
    if scope is None:
        return f"no recorded call for scope {stack!r}"
    depth = len(stack)
    for site in trace.eval_sites:
        if site[:depth] == stack:
            return "the scope evaluates mid-call (data-dependent control flow)"
    try:
        nodes = scope_nodes(trace, scope)
    except NotReplayable as e:
        return str(e)
    for node in nodes:
        if node.op == OPAQUE_OP:
            return "the scope contains a compiled call the harness cannot name"
        for out in node.out_arrays:
            if trace.liveness[out].kind is Retention.PYTHON_RETAINED:
                return (
                    f"the scope produces a python-retained value "
                    f"({node.op!r} at seq {node.seq}); the wrapper can never "
                    f"reach the reference the model kept"
                )
    try:
        emit_wrapper(trace, scope, [], "_ScreenProbe")
    except NotReplayable as e:
        return str(e)
    return None


@dataclass
class CertificationResult:
    ok: bool
    reason: str = ""


def certify_identity(
    build_wrapper: Callable[[], object],
    install: Callable[[object], Callable[[], None]],
    runs: Sequence[Callable[[], object]],
    repeated_calls: int = 3,
) -> CertificationResult:
    """Generic identity certification: capture baseline outputs of every run,
    install the identity wrapper, re-run, compare bitwise, uninstall.

    runs are zero-arg callables that each execute the model once on one
    workload and return the output tree; they are called repeated_calls times
    per arm so a stateful step must match call by call.
    """
    baseline: list[list] = []
    for run in runs:
        baseline.append([_flatten(run()) for _ in range(repeated_calls)])

    wrapper = build_wrapper()
    uninstall = install(wrapper)
    try:
        for i, run in enumerate(runs):
            for j in range(repeated_calls):
                got = _flatten(run())
                want = baseline[i][j]
                if len(got) != len(want):
                    return CertificationResult(
                        False, f"run {i} call {j}: output arity changed"
                    )
                for k, (g, w) in enumerate(zip(got, want)):
                    if not mx.array_equal(g, w).item():
                        return CertificationResult(
                            False,
                            f"run {i} call {j} output {k}: identity replay is not "
                            f"bitwise invisible",
                        )
    finally:
        uninstall()
    return CertificationResult(True)


def _flatten(tree: object) -> list[mx.array]:
    out = flatten_arrays(tree)
    mx.eval(out)
    return out
