"""Serve shipped kernels at every size of one dimension, after the search.

The search optimizes one recorded size. For `serve: {dim: [lo, hi]}` the job
also records the model at the ends and middle of that range, and then, for
each scope carrying shipped kernels:

1. Emits the scope's replay at every recorded size and turns the integers
   that change with the size into expressions of it (bind/sized.py). A scope
   whose replays differ in any other way keeps its searched delivery.
2. Checks the generalized replay against the original module at every size in
   the range: both are built as MLX graphs from the same arguments, and the
   graphs must be the same operations, attributes, constants and sharing,
   outputs and state alike (no GPU work; kernels off).
3. Checks each bit-exact kernel against its recorded operations at every size
   in the sandbox (shader validation, poisoned outputs, repeated runs).
4. Times each kernel against its operations on a grid of sizes, then at
   random sizes inside the range the grid picked; a kernel is served over its
   correct sizes around the optimized one, out to the last grid point on each
   side before one it does not win at, and never past a size it lost at.

The scope's generated class then replaces its searched wrapper: sizes in the
scope's range run the replay, and each kernel runs only inside its own range.
Everything else passes to the original module.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field

import mlx.core as mx

from autotuner_runtime import graph_native, kernels as runtime_kernels
from autotuner_runtime.graph import _is_holder, _python_snapshot, _rewind_python, _state_arrays
from autotuner_runtime.swap import ReplayWrapper, flatten_arrays

from .bind.emit import NotReplayable, Splice, scope_nodes
from .bind.sized import NotSizeGeneric, SizedWrapper, _fit, sized_wrapper
from .trace.serialize import node_to_dict


@dataclass
class ScopePlan:
    path: str
    workload: str
    address: str
    sized: SizedWrapper
    splices: list[Splice]              # at the primary size
    kernel_shapes: dict                # kernel id -> per input [[dim or [a, b]], dtype], a * n + b
    checked: dict = field(default_factory=dict)   # size -> None (same calculation) or why not
    to_dim: tuple | None = None        # (a, b): the manifest dim is a * size + b

    def span(self) -> tuple[int, int] | None:
        """The contiguous run of checked sizes around the primary size."""
        primary = self.sized.sizes[0]
        if self.checked.get(primary, "unchecked") is not None:
            return None
        lo = hi = primary
        while self.checked.get(lo - 1, "unchecked") is None:
            lo -= 1
        while self.checked.get(hi + 1, "unchecked") is None:
            hi += 1
        return lo, hi


def serve_points(lo: int, hi: int, primary: int) -> list[int]:
    """Where the model is recorded for generalizing: both ends and the middle."""
    return [n for n in dict.fromkeys((lo, (lo + hi) // 2, hi)) if n != primary]


def _corresponding(primary_trace, primary_scope, trace, scope):
    """Array ids and seqs of one scope call in another recording, matched
    node by node; the op sequence must be the same."""
    a, b = scope_nodes(primary_trace, primary_scope), scope_nodes(trace, scope)
    if len(a) != len(b) or any(x.op != y.op for x, y in zip(a, b)):
        raise NotSizeGeneric("the scope runs different operations at another size")
    ids, seqs = {}, {}
    for x, y in zip(a, b):
        seqs[x.seq] = y.seq
        for p, q in zip(x.in_arrays + x.out_arrays, y.in_arrays + y.out_arrays):
            if ids.setdefault(p, q) != q:
                raise NotSizeGeneric("the scope's dataflow differs at another size")
    return ids, seqs


def _calls(node_dicts) -> list:
    return [(d["op"], d["args"], d["kwargs"], d["kernel_definition"]) for d in node_dicts]


def plan_scopes(installed: dict, traces: dict, serve_traces: dict, workloads, dims) -> tuple[list, dict]:
    """A plan per scope that generalizes; why the others do not. dims: the
    manifest dim's value at the primary recording and at each serve trace."""
    plans, refused = [], {}
    for path, (_original, variants, _kernels) in installed.items():
        reasons = []
        for (workload, address), splices in variants.items():
            if not splices or workload not in workloads:
                continue
            trace = traces[workload]
            scope = next(sc for sc in trace.scope_calls if sc.address == address)
            recorded = [(trace, scope, splices)]
            try:
                for other in serve_traces[workload]:
                    other_scope = next((sc for sc in other.scope_calls if sc.address == address), None)
                    if other_scope is None:
                        raise NotSizeGeneric("the scope is not called at every recorded size")
                    ids, seqs = _corresponding(trace, scope, other, other_scope)
                    recorded.append((other, other_scope, [
                        Splice(s.kernel, seqs[s.start_seq], seqs[s.end_seq],
                               tuple(ids[a] for a in s.input_ids), tuple(ids[a] for a in s.output_ids),
                               s.fingerprint) for s in splices]))
                sized = sized_wrapper(recorded, path)
            except (NotSizeGeneric, NotReplayable) as error:
                reasons.append(f"{address}: {error}")
                continue
            plans.append(ScopePlan(path, workload, address, sized, list(splices),
                                   _kernel_shapes(recorded, sized.sizes), to_dim=_fit(dims, sized.sizes)))
            break
        else:
            refused[path] = "; ".join(reasons) or "no shipped kernel at a served workload"
    return plans, refused


def _kernel_shapes(recorded, sizes) -> dict:
    """Each kernel's input shapes as exact integer functions of the scope's
    size, when its recorded operations take no size-dependent settings (the
    sandbox check replays those operations at every size)."""
    shapes = {}
    for i, splice in enumerate(recorded[0][2]):
        per_size, calls = [], []
        for trace, _scope, splices in recorded:
            s = splices[i]
            specs = trace.span_specs(s.start_seq, s.end_seq)
            per_size.append([specs[a] for a in s.input_ids])
            calls.append(_calls(node_to_dict(n) for n in trace.nodes[s.start_seq:s.end_seq + 1]))
        kid = splice.kernel.kernel_id
        if any(c != calls[0] for c in calls):
            shapes[kid] = None  # its operations take a size-dependent setting
            continue
        template = []
        for j, (shape, dtype) in enumerate(per_size[0]):
            dims = []
            for d in range(len(shape)):
                values = [p[j][0][d] for p in per_size]
                fit = values[0] if len(set(values)) == 1 else _fit(values, sizes)
                dims.append(list(fit) if isinstance(fit, tuple) else fit)
            if None in dims or any(len(p[j][0]) != len(shape) or p[j][1] != dtype for p in per_size):
                template = None
                break
            template.append([dims, dtype])
        if shapes.get(kid, template) != template:
            template = None  # two scopes size one kernel differently
        shapes[kid] = template
    return shapes


# -- 2: the replay against the original at every size ------------------------

class _Enough(Exception):
    """Every scope was checked on this pass; the rest of the model need not run."""


@contextlib.contextmanager
def _kernels_off():
    saved = runtime_kernels.try_sized, runtime_kernels.try_call
    runtime_kernels.try_sized = runtime_kernels.try_call = lambda spec, inputs: None
    try:
        yield
    finally:
        runtime_kernels.try_sized, runtime_kernels.try_call = saved


def _holders(value, out):
    if isinstance(value, (tuple, list)):
        for v in value:
            _holders(v, out)
    elif isinstance(value, dict):
        for v in value.values():
            _holders(v, out)
    elif _is_holder(value) and value not in out:
        out.append(value)
    return out


def size_reader(sized: SizedWrapper):
    """The generated wrapper's own size read, as a function of its arguments."""
    match = re.search(r"^    def __call__\(self(?:, )?(?P<sig>[^\n]*)\):\n        (?P<read>_n = [^\n]*)\n",
                      sized.template, re.M)
    namespace = {"mx": mx}
    exec(f"def size({match['sig']}):\n    {match['read']}\n    return _n\n", namespace)
    return namespace["size"]


class _Pass:
    """One run of the model: which scopes were checked, and a stop once all were."""

    def __init__(self, paths):
        self.waiting = set(paths)

    def reached(self, path):
        self.waiting.discard(path)
        if not self.waiting:
            raise _Enough


class _Check(ReplayWrapper):
    """Stands in for one scope during the check: builds the candidate's graph
    and the original's from the same arguments, compares them, and returns
    the original's result so the model runs on as before."""

    def __init__(self, original, candidate, plan: ScopePlan, bounds):
        super().__init__(original)
        object.__setattr__(self, "_check", (candidate, plan, size_reader(plan.sized), bounds))
        object.__setattr__(self, "run", None)

    def __call__(self, *args, **kwargs):
        candidate, plan, size, bounds = self._check
        n = size(*args, **kwargs)
        if not bounds[0] <= n <= bounds[1]:
            return self.wrapped(*args, **kwargs)
        if n not in plan.checked:
            holders = _holders((args, kwargs), [])
            saved = _python_snapshot((args, kwargs))
            with _kernels_off():
                ours = flatten_arrays(candidate(*args, **kwargs)) + _state_arrays([vars(h) for h in holders])
            _rewind_python(saved)
            result = self.wrapped(*args, **kwargs)
            theirs = flatten_arrays(result) + _state_arrays([vars(h) for h in holders])
            same, why = graph_native.same_computation(theirs, ours)
            plan.checked[n] = None if same else why
        else:
            result = self.wrapped(*args, **kwargs)
        self.run.reached(plan.path)
        return result


def check_replays(model, plans: list[ScopePlan], candidates: dict, run_at, sizes, install, uninstall):
    """Drive the model once per size with every planned scope checked in
    place, stopping each run once every scope was reached (nothing is
    evaluated by then). run_at(size) runs the model on inputs of that size;
    install and uninstall swap modules at a path."""
    checks, occupants = [], []
    for plan in plans:
        check = _Check(candidates[plan.path].wrapped, candidates[plan.path], plan,
                       (min(plan.sized.sizes), max(plan.sized.sizes)))
        occupants.append((plan.path, install(model, plan.path, check)))
        checks.append(check)
    try:
        for size in sizes:
            run = _Pass(plan.path for plan in plans)
            for check in checks:
                object.__setattr__(check, "run", run)
            try:
                run_at(size)
            except _Enough:
                pass
    finally:
        for path, occupant in reversed(occupants):
            uninstall(model, path, occupant)


# -- 4: where a kernel is served -----------------------------------------------

def served_range(correct: list[int], clock: dict[int, dict], primary: int,
                 grid=None) -> tuple[int, int] | None:
    """The kernel's contiguous correct sizes around the primary, out on each
    side to the furthest timed size it wins at, stopping at a size it is
    clearly slower at, or at a grid point it does not win at. clock: size ->
    {"win", "loss"} for the grid and the random samples; a sample that only
    ties neither extends nor stops the range. The primary size was already
    accepted, so it counts as a win."""
    grid = set(clock) if grid is None else set(grid)
    ok = set(correct) - {n for n, row in clock.items() if row.get("loss")}
    if primary not in ok:
        return None
    lo = hi = primary
    while lo - 1 in ok:
        lo -= 1
    while hi + 1 in ok:
        hi += 1
    ends = []
    for points in (sorted((n for n in clock if lo <= n < primary), reverse=True),
                   sorted(n for n in clock if primary < n <= hi)):
        end = primary
        for n in points:
            if n in grid and not clock[n].get("win"):
                break
            if clock[n].get("win"):
                end = n
        ends.append(end)
    return ends[0], ends[1]


def in_dim(plan_or_map, span):
    """A range of a scope's size in the manifest dim's units."""
    fit = plan_or_map.to_dim if isinstance(plan_or_map, ScopePlan) else plan_or_map
    if span is None or fit is None:
        return None
    a, b = fit
    return sorted((a * span[0] + b, a * span[1] + b))
