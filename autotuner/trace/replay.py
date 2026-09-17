"""Replay: re-execute a recorded node sequence.

Four consumers share this one module: region-clock pricing, the fp32 golden,
sweep reference generation, and the wrapper generator (which emits the same op
names as source). Ops resolve through the op table's saved originals, so
replay is patch-invisible in a traced process and works identically in a
patch-free subprocess.

Replaying a mutating op (__setitem__, an in-place dunder) mutates the bound
array object, exactly as the model did; callers who must keep their bindings
pristine pass fresh copies (boundary stores load fresh tensors per replay).
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import mlx.core as mx

from autotuner_runtime.kernels import imported

from .optable import MUTATING_METHODS, resolve
from autotuner_runtime.captured_kernels import captured, definition_key
from .recorder import OPAQUE_OP, UNNAMED_KERNEL_OP, ArrayRef, compiled_path, kernel_path, state_method
from .types import TraceNode
from .walk import flatten_arrays


def _materialize(obj: object, arrays: Sequence[mx.array]) -> object:
    if isinstance(obj, ArrayRef):
        return arrays[obj.index]
    if isinstance(obj, tuple):
        return tuple(_materialize(v, arrays) for v in obj)
    if isinstance(obj, list):
        return [_materialize(v, arrays) for v in obj]
    if isinstance(obj, dict):
        return {k: _materialize(v, arrays) for k, v in obj.items()}
    if isinstance(obj, slice):
        return slice(
            _materialize(obj.start, arrays),
            _materialize(obj.stop, arrays),
            _materialize(obj.step, arrays),
        )
    return obj


def _collect_outputs(node: TraceNode, args: tuple, result: object) -> list[mx.array]:
    mutated = [args[0]] if node.op.removeprefix("array.") in MUTATING_METHODS else []
    return mutated + flatten_arrays(result)


def replay(
    nodes: Sequence[TraceNode],
    bindings: Mapping[int, mx.array],
    outputs: Iterable[int],
    op_substitute: Mapping[str, Callable] | None = None,
) -> dict[int, mx.array]:
    """Fold over nodes in seq order, binding array_ids to live arrays. bindings
    must cover every array the span reads from outside (inputs and weights);
    returns live arrays for the requested output ids."""
    env: dict[int, mx.array] = dict(bindings)
    for node in nodes:
        if op_substitute and node.op in op_substitute:
            fn = op_substitute[node.op]
        elif node.kernel_definition is not None:
            fn = captured(definition_key(node.kernel_definition))
        elif compiled_path(node.op) or kernel_path(node.op):
            fn = imported(compiled_path(node.op) or kernel_path(node.op))
        elif node.op == OPAQUE_OP:
            raise RuntimeError(f"seq {node.seq}: a compiled call with no import path cannot be replayed")
        elif node.op == UNNAMED_KERNEL_OP:
            raise RuntimeError(f"seq {node.seq}: a custom kernel with no import path cannot be replayed")
        elif state_method(node.op):
            raise RuntimeError(f"seq {node.seq}: {node.op} acts on the model's own state; only a "
                               f"generated wrapper replays it, on the live object")
        else:
            fn = resolve(node.op)
        bound: list[mx.array] = []
        for aid in node.in_arrays:
            if aid not in env:
                raise KeyError(
                    f"replay of {node.op!r} (seq {node.seq}) needs array {aid} "
                    f"which is neither bound nor produced earlier in the span"
                )
            bound.append(env[aid])
        args = tuple(_materialize(a, bound) for a in node.scalar_args["args"])
        kwargs = {k: _materialize(v, bound) for k, v in node.scalar_args["kwargs"].items()}
        result = fn(*args, **kwargs)
        outs = _collect_outputs(node, args, result)
        if len(outs) != len(node.out_arrays):
            raise RuntimeError(
                f"replay of {node.op!r} (seq {node.seq}) produced {len(outs)} arrays, "
                f"recorded {len(node.out_arrays)}"
            )
        for aid, arr in zip(node.out_arrays, outs):
            env[aid] = arr
    return {aid: env[aid] for aid in outputs}


def compile_replay(nodes: Sequence[TraceNode], input_ids: Iterable[int],
                   output_ids: Iterable[int],
                   fixed_bindings: Mapping[int, mx.array] | None = None) -> Callable:
    """Build the compiled region arm shared by pricing and candidate clocks.

    Replay once, then collect outputs in boundary order. Replaying per output
    duplicates shared custom dispatches even when their values are identical.
    Callers own warmup and pacing; constructing this callable runs no GPU work.
    Per-call inputs override fixed bindings, as in ordinary region replay.
    """
    ids, outputs = tuple(sorted(input_ids)), tuple(output_ids)
    fixed = dict(fixed_bindings or {})

    @mx.compile
    def compiled(*arrays):
        values = replay(nodes, {**fixed, **dict(zip(ids, arrays))}, outputs)
        return [values[aid] for aid in outputs]

    def run(bindings):
        return compiled(*[bindings[aid] for aid in ids])

    return run


def prepare_replay(nodes: Sequence[TraceNode], input_ids: Iterable[int],
                   output_ids: Iterable[int]) -> Callable:
    """Resolve a span once into ordinary Python calls for the plain clock.

    This does not use mx.compile: MLX still executes the original operations.
    Only trace interpretation (op lookup and argument reconstruction) moves
    outside the timer. Constants are bound objects, never interpolated code.
    """
    namespace = {"_outputs": _prepared_outputs}
    names = {aid: f"v{i}" for i, aid in enumerate(dict.fromkeys(input_ids))}
    lines = ["def run(bindings):"]
    lines += [f"    {name} = bindings[{aid!r}]" for aid, name in names.items()]

    def constant(value):
        name = f"c{len(namespace)}"
        namespace[name] = value
        return name

    def expression(value, bound):
        if isinstance(value, ArrayRef):
            return bound[value.index]
        if isinstance(value, tuple):
            return "(" + "".join(expression(v, bound) + "," for v in value) + ")"
        if isinstance(value, list):
            return "[" + ",".join(expression(v, bound) for v in value) + "]"
        if isinstance(value, dict):
            return "{" + ",".join(constant(k) + ":" + expression(v, bound)
                                   for k, v in value.items()) + "}"
        if isinstance(value, slice):
            return constant(slice) + "(" + ",".join(expression(v, bound) for v in
                                                      (value.start, value.stop, value.step)) + ")"
        return constant(value)

    for index, node in enumerate(nodes):
        if (node.op == OPAQUE_OP or (node.op == UNNAMED_KERNEL_OP and node.kernel_definition is None)
                or state_method(node.op)):
            raise RuntimeError(f"{node.op} cannot be replayed as a plain region")
        path = compiled_path(node.op) or kernel_path(node.op)
        fn = (captured(definition_key(node.kernel_definition)) if node.kernel_definition is not None
              else imported(path) if path else resolve(node.op))
        bound = [names[aid] for aid in node.in_arrays]
        args = [expression(v, bound) for v in node.scalar_args["args"]]
        kwargs = node.scalar_args["kwargs"]
        # ** preserves arbitrary keyword names without emitting them as source.
        arguments = ",".join(args + (["**" + expression(kwargs, bound)] if kwargs else []))
        lines.append(f"    r{index} = {constant(fn)}({arguments})")
        mutated = node.op.removeprefix("array.") in MUTATING_METHODS
        output_names = [f"o{index}_{i}" for i in range(len(node.out_arrays))]
        if output_names:
            lhs = ",".join(output_names) + ","
            prefix = f"[{args[0]}]" if mutated else "None"
            lines.append(f"    {lhs} = _outputs(r{index}, {prefix})")
        else:
            lines.append(f"    if _outputs(r{index}, None): raise RuntimeError('unexpected replay outputs')")
        names.update(zip(node.out_arrays, output_names))
    lines.append("    return [" + ",".join(names[aid] for aid in output_ids) + "]")
    exec(compile("\n".join(lines), "<prepared-region>", "exec"), namespace)
    return namespace["run"]


def _prepared_outputs(result, mutated):
    arrays = [result] if isinstance(result, mx.array) else flatten_arrays(result)
    return mutated + arrays if mutated else arrays
