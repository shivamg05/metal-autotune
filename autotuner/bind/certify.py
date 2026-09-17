"""The static replayability screen and identity certification.

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
import mlx.nn as nn

from ..trace.recorder import OPAQUE_OP, UNNAMED_KERNEL_OP
from ..trace.types import Retention, ScopeCall, Trace
from ..trace.walk import flatten_arrays
from ..artifact.validate import _structure
from .emit import MODULE_HEADER, NotReplayable, emit_wrapper, scope_nodes
from autotuner_runtime.exact import bitwise_equal
from autotuner_runtime.swap import ReplayWrapper, install as swap_install, uninstall as swap_uninstall, resolve


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
        if node.op == UNNAMED_KERNEL_OP and node.kernel_definition is None:
            return "the scope contains a custom kernel the harness cannot name"
        for out in node.out_arrays:
            if trace.liveness[out].kind is Retention.PYTHON_RETAINED:
                return (
                    f"the scope produces a python-retained value "
                    f"({node.op!r} at seq {node.seq}); the wrapper can never "
                    f"reach the reference the model kept"
                )
    try:
        emitted = emit_wrapper(trace, scope, [], "_ScreenProbe")
        compile(MODULE_HEADER + emitted.source, "<screen>", "exec")
    except NotReplayable as e:
        return str(e)
    except SyntaxError as e:
        return f"the generated wrapper is not valid Python: {e.msg} at line {e.lineno}"
    return None


@dataclass
class CertificationResult:
    ok: bool
    reason: str = ""
    scope: str | None = None   # the scope whose identity failed, when one is known


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
                            f"run {i} call {j} output {k}: identity replay is not bitwise "
                            f"invisible (a step whose state moves between calls, such as a "
                            f"cache offset, must be rewound by the model file so every call "
                            f"is the same step)",
                        )
    finally:
        uninstall()
    return CertificationResult(True)


def _flatten(tree: object) -> list[mx.array]:
    out = flatten_arrays(tree)
    mx.eval(out)
    return out


def certify_identities(model, wrappers: dict[str, object], runs: Sequence[Callable],
                       repeated_calls: int = 3) -> CertificationResult:
    """Check a batch of independent scopes in six model passes per workload.

    Observe every scope's outputs in both arms, so two wrong wrappers cannot
    pass by cancelling at the model output. Nested targets run in separate
    batches: a parent's replay may bypass its children. All temporary swaps
    unwind even when a call or comparison raises. Runs must rewind model state,
    as they do for the other whole-model correctness checks.
    """
    pending = dict(wrappers)
    if repeated_calls < 1:
        raise ValueError("identity certification requires at least one call per arm")
    while pending:
        paths = [p for p in pending if not any(p.startswith(q + ".") for q in pending if q != p)]
        batch = {p: pending.pop(p) for p in paths}
        result = _certify_batch(model, batch, runs, repeated_calls)
        if not result.ok:
            return result
    return CertificationResult(True)


def _certify_batch(model, wrappers, runs, repeated_calls):
    def snapshot(out):
        # Preserve scalar state and container structure as well as array values.
        return _structure(out), [mx.array(a) for a in flatten_arrays(out)]

    class Observed(ReplayWrapper):
        def __init__(self, wrapped, path, observations, order):
            nn.Module.__init__(self)
            self.wrapped = wrapped
            object.__setattr__(self, "_path", path)
            object.__setattr__(self, "_observations", observations)
            object.__setattr__(self, "_order", order)

        def __call__(self, *args, **kwargs):
            self._order.setdefault(self._path, len(self._order))
            out = self.wrapped(*args, **kwargs)
            # Snapshot handles before the model can update an array in place.
            self._observations[self._path].append(snapshot(out))
            return out

    def collect(run, replacement):
        observations = {path: [] for path in wrappers}
        order = {}  # path -> position of its first call: a changed output blames the earliest scope
        installed = []
        try:
            for path in wrappers:
                original = resolve(model, path)[2]
                target = wrappers[path] if replacement else original
                previous = swap_install(model, path, Observed(target, path, observations, order))
                installed.append((path, previous))
            observations[""] = [snapshot(run())]
            order[""] = len(order)
            mx.eval([values for calls in observations.values() for _, values in calls])
            return observations, order
        finally:
            for path, original in reversed(installed):
                swap_uninstall(model, path, original)

    seen = set()
    for index, run in enumerate(runs):
        # Keep only one pair's activations, rather than three full passes.
        for repetition in range(repeated_calls):
            (expected, _), (actual, order) = collect(run, False), collect(run, True)
            # A scope whose compiled output differs changes every scope after
            # it, so the mismatch that executed first is the one to blame.
            for path in sorted(expected, key=lambda p: order.get(p, len(order))):
                if expected[path]:
                    seen.add(path)
                if len(expected[path]) != len(actual[path]):
                    return CertificationResult(False, f"identity scope {path!r}: call count changed at workload {index}", path)
                for call_index, (want, got) in enumerate(zip(expected[path], actual[path])):
                    want_structure, want_values = want
                    got_structure, got_values = got
                    if (want_structure != got_structure or len(want_values) != len(got_values)
                            or any(not bitwise_equal(a, b) for a, b in zip(want_values, got_values))):
                        return CertificationResult(False, f"identity scope {path or '<model>'!r}: output changed "
                                                   f"at workload {index}, repetition {repetition}, call {call_index}",
                                                   path or None)
            del expected, actual
    missing = set(wrappers) - seen
    if missing:
        return CertificationResult(False, f"identity scopes never executed: {sorted(missing)}", sorted(missing)[0])
    return CertificationResult(True)
