"""Record mode: node capture, array identity, module addressing.

The recorder holds a strong reference to every array it sees, so id() is a
sound identity for the whole pass. Mutation (x[i] = v, an in-place dunder
returning its own receiver, an op returning an input object) is handled as SSA
renaming: the mutated object gets a fresh array_id going forward, produced by
the mutating node; earlier consumers keep the old id.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import mlx.core as mx

from .freeze import freeze
from .types import ScopeCall, Trace, TraceNode
from .walk import flatten_arrays, snapshot_arrays

OPAQUE_OP = "compiled_fn"       # a compiled call the harness cannot name
COMPILED_PREFIX = "compiled:"   # a compiled call named by its import path
STATE_PREFIX = "state:"         # a method call on an object that holds model state


def compiled_op(path: str | None) -> str:
    return f"{COMPILED_PREFIX}{path}" if path else OPAQUE_OP


def is_opaque(op: str) -> bool:
    """A compiled section or a state call: never inside a region, always a
    chain barrier."""
    return op == OPAQUE_OP or op.startswith(COMPILED_PREFIX) or op.startswith(STATE_PREFIX)


def compiled_path(op: str) -> str | None:
    return op[len(COMPILED_PREFIX):] if op.startswith(COMPILED_PREFIX) else None


def state_method(op: str) -> str | None:
    """The method name of a state call, e.g. "update_and_fetch" for
    "state:KVCache.update_and_fetch"; None for any other op."""
    return op.rsplit(".", 1)[1] if op.startswith(STATE_PREFIX) else None


@dataclass(frozen=True)
class ArrayRef:
    """Placeholder for an array argument inside a node's args template."""
    index: int


@dataclass(frozen=True)
class ObjectRef:
    """Placeholder for a non-array, non-literal argument of a module call
    (a cache object handed down the tree); index into the scope's obj_ids."""
    index: int


def _is_literal(obj: object) -> bool:
    return obj is None or obj is Ellipsis or isinstance(obj, (bool, int, float, str, mx.Dtype))


def dtype_name(dtype: mx.Dtype) -> str:
    return str(dtype).removeprefix("mlx.core.")


def _spec(arr: mx.array) -> tuple[tuple[int, ...], str]:
    return (tuple(arr.shape), dtype_name(arr.dtype))


class Recorder:
    # Snapshot walks skip anything flagged _trace_internal, so a model holding
    # a compiled proxy (which references this recorder) cannot leak the
    # recorder's own held arrays into weights or retention.
    _trace_internal = True

    def __init__(self) -> None:
        self.armed = False
        self._suppress = 0
        self._model_for_walk: object | None = None
        # id(object) -> path for the objects holding model state; the patcher
        # sets it when it wraps the model, before any pass
        self.state_holders: dict[int, str] = {}
        self._reset_pass()

    def _reset_pass(self) -> None:
        self.nodes: list[TraceNode] = []
        self._holds: list[mx.array] = []
        self._ids: dict[int, int] = {}
        self._by_aid: dict[int, mx.array] = {}
        self._next_id = 0
        self.inputs: set[int] = set()
        self.weights: set[int] = set()
        self.weight_paths: dict[int, str] = {}
        self.step_outputs: list[int] = []
        self.in_pass_evaluation = False
        self.eval_sites: list[tuple[str, ...]] = []
        self._addr_stack: list[str] = []
        self._pending_scopes: list[tuple] = []
        self.scope_calls: list[ScopeCall] = []
        self._instance_paths: dict[int, str] = {}
        self._call_counts: dict[int, int] = {}
        self._pos_counts: dict[tuple[str, str], int] = {}
        self._root_id: int | None = None

    @property
    def recording(self) -> bool:
        return self.armed and self._suppress == 0

    @contextmanager
    def suppressed(self) -> Iterator[None]:
        """Silence recording inside an opaque call (a compiled section's trace
        re-executes its python; those inner ops are not part of the record)."""
        self._suppress += 1
        try:
            yield
        finally:
            self._suppress -= 1

    def _register(self, arr: mx.array) -> int:
        oid = id(arr)
        if oid in self._ids:
            return self._ids[oid]
        self._holds.append(arr)
        self._ids[oid] = self._next_id
        self._by_aid[self._next_id] = arr
        self._next_id += 1
        return self._ids[oid]

    def _register_output(self, arr: mx.array) -> int:
        """An output object already known under an id was mutated (or returned
        unchanged): rename so the node is its producer from here on."""
        oid = id(arr)
        if oid in self._ids:
            self._holds.append(arr)
            self._ids[oid] = self._next_id
            self._by_aid[self._next_id] = arr
            self._next_id += 1
            return self._ids[oid]
        return self._register(arr)

    def arrays_for(self, ids: Iterable[int]) -> dict[int, mx.array]:
        """The live (possibly lazy) arrays for array_ids, valid until the holds
        drop at freeze. Boundary capture evaluates and saves these."""
        return {aid: self._by_aid[aid] for aid in ids}

    # -- pass lifecycle ------------------------------------------------------

    def arm(self, model: object, workload_arrays: list[mx.array], module_paths: dict[int, str]) -> None:
        """Start a recording pass: register weights (snapshot walk) and inputs,
        reset all per-pass state. module_paths maps id(instance) -> tree path."""
        if self.armed:
            raise RuntimeError("recorder already armed")
        self._reset_pass()
        self._register_reachable(model)
        self._instance_paths = dict(module_paths)
        self._root_id = id(model)
        self._model_for_walk = model
        for arr in workload_arrays:
            self.inputs.add(self._register(arr))
        self.armed = True

    def _register_reachable(self, model: object) -> None:
        paths = snapshot_arrays(model)
        found: list[mx.array] = []
        _collect_objects(model, found)
        for arr in found:
            aid = self._register(arr)
            self.weights.add(aid)
            self.weight_paths[aid] = paths.get(id(arr), "?")

    def step(self, model: Callable, args: tuple) -> object:
        """One traced call of the model: roots the address space, resets the
        per-step counters, captures the returned tree."""
        if not self.armed:
            raise RuntimeError("recorder is not armed")
        self._call_counts.clear()
        self._pos_counts.clear()
        k = self._call_counts.get(id(model), 0)
        self._call_counts[id(model)] = k + 1
        self._addr_stack.append(f"@{k}")
        args_t, kwargs_t, arg_ids, _, obj_ids = self._templatize(args, {}, objects=True)
        try:
            outs = model(*args)
        except BaseException:
            self._addr_stack.pop()
            raise
        out_t, _, out_ids, _, _ = self._templatize((outs,), {})
        self.scope_calls.append(ScopeCall(
            address=self._addr_stack[-1], stack=tuple(self._addr_stack),
            args_template=args_t, kwargs_template=kwargs_t, arg_ids=tuple(arg_ids),
            out_template=out_t[0], out_ids=tuple(out_ids), obj_ids=tuple(obj_ids),
        ))
        self._addr_stack.pop()
        self.step_outputs = [self._register(a) for a in flatten_arrays(outs)]
        return outs

    def disarm(self) -> None:
        self.armed = False

    def freeze_pass(self) -> Trace:
        """Freeze after a step: retention via the positive snapshot walk (any
        produced array reachable from the model), then drop everything."""
        produced = {aid for n in self.nodes for aid in n.out_arrays}
        post = snapshot_arrays(self._model_for_walk) if self._model_for_walk else {}
        retained = {self._ids[oid] for oid in post if oid in self._ids} & produced
        trace = freeze(
            self.nodes,
            inputs=self.inputs,
            weights=self.weights,
            step_outputs=self.step_outputs,
            retained=retained,
            weight_paths=self.weight_paths,
            in_pass_evaluation=self.in_pass_evaluation,
            eval_sites=tuple(self.eval_sites),
            scope_calls=tuple(self.scope_calls),
        )
        self._holds.clear()
        self._by_aid.clear()
        return trace

    # -- hooks called by the patch surface -----------------------------------

    def note_evaluation(self) -> None:
        if self.recording:
            self.in_pass_evaluation = True
            self.eval_sites.append(tuple(self._addr_stack))

    def module_enter(self, instance: object, args: tuple = (), kwargs: dict | None = None) -> None:
        path = self._instance_paths.get(id(instance), f"?{type(instance).__name__}")
        k = self._call_counts.get(id(instance), 0)
        self._call_counts[id(instance)] = k + 1
        self._addr_stack.append(f"{path}@{k}")
        args_t, kwargs_t, ids, _, obj_ids = self._templatize(args, kwargs or {}, objects=True)
        self._pending_scopes.append((self._addr_stack[-1], tuple(self._addr_stack), args_t, kwargs_t,
                                     tuple(ids), tuple(obj_ids)))

    def module_exit(self, result: object = None) -> None:
        address, stack, args_t, kwargs_t, arg_ids, obj_ids = self._pending_scopes.pop()
        out_t, _, out_ids, _, _ = self._templatize((result,), {})
        self.scope_calls.append(ScopeCall(
            address=address, stack=stack, args_template=args_t,
            kwargs_template=kwargs_t, arg_ids=arg_ids,
            out_template=out_t[0], out_ids=tuple(out_ids), obj_ids=obj_ids,
        ))
        self._addr_stack.pop()

    def module_abort(self) -> None:
        self._pending_scopes.pop()
        self._addr_stack.pop()

    def is_root(self, instance: object) -> bool:
        return id(instance) == self._root_id

    def _templatize(self, args: tuple, kwargs: dict, objects: bool = False):
        """Arrays become ArrayRefs. With objects=True (module calls), any
        other non-literal value becomes an ObjectRef, so a wrapper can pass
        a cache object through and call its methods where the record did."""
        in_ids: list[int] = []
        in_specs: list = []
        obj_ids: list[int] = []

        def template(obj: Any) -> Any:
            if isinstance(obj, mx.array):
                in_ids.append(self._register(obj))
                in_specs.append(_spec(obj))
                return ArrayRef(len(in_ids) - 1)
            if isinstance(obj, tuple):
                return tuple(template(v) for v in obj)
            if isinstance(obj, list):
                return [template(v) for v in obj]
            if isinstance(obj, dict):
                return {k: template(v) for k, v in obj.items()}
            if isinstance(obj, slice):
                if any(isinstance(v, mx.array) for v in (obj.start, obj.stop, obj.step)):
                    return slice(template(obj.start), template(obj.stop), template(obj.step))
                return obj
            if objects and not _is_literal(obj):
                obj_ids.append(id(obj))
                return ObjectRef(len(obj_ids) - 1)
            return obj

        args_t = tuple(template(a) for a in args)
        kwargs_t = {k: template(v) for k, v in kwargs.items()}
        return args_t, kwargs_t, in_ids, in_specs, obj_ids

    def maybe_record(
        self,
        op_name: str,
        args: tuple,
        kwargs: dict,
        result: Any,
        mutates_first: bool = False,
    ) -> None:
        if not self.recording:
            return
        out_objs = ([args[0]] if mutates_first else []) + flatten_arrays(result)
        if not out_objs:
            return
        self._append_node(op_name, args, kwargs, out_objs)

    def record_state_call(self, obj: object, method: str, args: tuple, kwargs: dict, result: Any) -> None:
        """One node for a call on a state holder, outputs or not: the wrapper
        replays it by calling the same method on the same object, so what
        the method does to its state (a cache write, an offset bump) happens
        for real, where the recorder cannot see it."""
        if not self.recording:
            return
        op = f"{STATE_PREFIX}{type(obj).__name__}.{method}"
        receiver = {"id": id(obj), "path": self.state_holders.get(id(obj))}
        self._append_node(op, args, kwargs, flatten_arrays(result), receiver=receiver)

    def _append_node(self, op_name: str, args: tuple, kwargs: dict, out_objs: list,
                     receiver: dict | None = None) -> None:
        args_t, kwargs_t, in_ids, in_specs, _ = self._templatize(args, kwargs)
        out_ids = [self._register_output(a) for a in out_objs]
        scalar_args: dict = {"args": args_t, "kwargs": kwargs_t}
        if receiver is not None:
            scalar_args["receiver"] = receiver

        stack = tuple(self._addr_stack)
        address = stack[-1] if stack else ""
        pos_key = (address, op_name)
        position = self._pos_counts.get(pos_key, 0)
        self._pos_counts[pos_key] = position + 1

        self.nodes.append(TraceNode(
            seq=len(self.nodes),
            op=op_name,
            in_arrays=tuple(in_ids),
            out_arrays=tuple(out_ids),
            in_specs=tuple(in_specs),
            out_specs=tuple(_spec(a) for a in out_objs),
            scalar_args=scalar_args,
            module_address=address,
            position_in_module=position,
            module_stack=stack,
        ))


def _collect_objects(root: object, out: list[mx.array]) -> None:
    """Companion to snapshot_arrays that keeps the array objects themselves."""
    seen: set[int] = set()

    def visit(obj: object) -> None:
        if id(obj) in seen or getattr(obj, "_trace_internal", False):
            return
        seen.add(id(obj))
        if isinstance(obj, mx.array):
            out.append(obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                visit(v)
            if hasattr(obj, "__dict__"):
                for v in vars(obj).values():
                    visit(v)
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                visit(v)
            return
        if isinstance(obj, set):
            for v in obj:
                visit(v)
            return
        if callable(obj):
            closure = getattr(obj, "__closure__", None)
            if closure:
                for cell in closure:
                    visit(cell.cell_contents)
            for v in getattr(obj, "__globals__", {}).values():
                if isinstance(v, (mx.array, list, tuple, dict)) and not isinstance(v, type):
                    visit(v)
        if hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                visit(v)

    visit(root)
