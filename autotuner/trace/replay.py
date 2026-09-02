"""Replay: re-execute a recorded node sequence (plan 5.4).

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
from .recorder import OPAQUE_OP, ArrayRef, compiled_path
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
        elif compiled_path(node.op):
            fn = imported(compiled_path(node.op))
        elif node.op == OPAQUE_OP:
            raise RuntimeError(f"seq {node.seq}: a compiled call with no import path cannot be replayed")
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
