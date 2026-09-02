"""One walk over everything reachable from a model object, with dot paths.

It handles nn.Module dict storage, plain attributes, nested containers, and,
for plain-function models, closure cells and module globals. The recorder's
weights, the retention check, and the state holders all come from it, so
they agree on what is reachable and by which path.
"""

from __future__ import annotations

import inspect
from typing import Iterator

import mlx.core as mx

from autotuner_runtime.swap import flatten_arrays  # noqa: F401  (the tracer's callers import it from here)


def reachable(root: object) -> Iterator[tuple[str, object]]:
    """(dot path, object) for everything reachable from root, parents before
    children. Containers and objects appear once; arrays are leaves and
    appear under every path that reaches them, so an object's own arrays
    are always seen under the object."""
    seen: set[int] = set()

    def visit(obj: object, path: str) -> Iterator[tuple[str, object]]:
        if isinstance(obj, mx.array):
            yield path, obj
            return
        oid = id(obj)
        if oid in seen or getattr(obj, "_trace_internal", False):
            return
        seen.add(oid)
        yield path, obj
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                yield from visit(v, f"{path}.{k}" if path else str(k))
            # nn.Module subclasses dict; its plain attributes live in __dict__
            if hasattr(obj, "__dict__"):
                for k, v in list(vars(obj).items()):
                    yield from visit(v, f"{path}.{k}" if path else str(k))
            return
        if isinstance(obj, (list, tuple)):
            # dot style so weight paths compose with scope tree paths
            for i, v in enumerate(obj):
                yield from visit(v, f"{path}.{i}")
            return
        if isinstance(obj, set):
            for v in obj:
                yield from visit(v, f"{path}{{}}")
            return
        if callable(obj):
            closure = getattr(obj, "__closure__", None)
            if closure:
                for i, cell in enumerate(closure):
                    yield from visit(cell.cell_contents, f"{path}<cell{i}>")
            for k, v in getattr(obj, "__globals__", {}).items():
                if isinstance(v, (mx.array, list, tuple, dict)) and not isinstance(v, type):
                    yield from visit(v, f"{path}<global:{k}>")
        if hasattr(obj, "__dict__"):
            for k, v in list(vars(obj).items()):
                yield from visit(v, f"{path}.{k}" if path else str(k))

    yield from visit(root, "")


def snapshot_arrays(root: object) -> dict[int, str]:
    """id(array) -> the first dot path that reaches it."""
    found: dict[int, str] = {}
    for path, obj in reachable(root):
        if isinstance(obj, mx.array):
            found.setdefault(id(obj), path)
    return found


def state_holders(root: object) -> list[tuple[str, object]]:
    """(dot path, object) for every plain Python object reachable from root
    that holds an array: a KV cache, a namespace of buffers, a helper with a
    table. Modules, containers, arrays, and functions are not holders."""
    arrays = [path for path, obj in reachable(root) if isinstance(obj, mx.array)]
    holders = []
    for path, obj in reachable(root):
        if (isinstance(obj, (mx.array, dict, list, tuple, set)) or inspect.isroutine(obj)
                or not hasattr(obj, "__dict__")):
            continue
        prefix = f"{path}." if path else ""
        if any(a.startswith(prefix) for a in arrays):
            holders.append((path, obj))
    return holders
