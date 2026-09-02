"""Walks over arrays: every mx.array reachable from a model object with its
path, and every array in a nested output tree.

The snapshot walk runs twice per recorded pass: before arming, the reachable
arrays are the model's weights; after the pass, any recorded production found
reachable is python_retained. It handles nn.Module dict storage, plain
attributes, nested containers, and, for plain-function models, closure cells
and module globals.
"""

from __future__ import annotations

import mlx.core as mx


def snapshot_arrays(root: object) -> dict[int, str]:
    """id(array) -> dot path for every array reachable from root."""
    found: dict[int, str] = {}
    seen: set[int] = set()

    def visit(obj: object, path: str) -> None:
        oid = id(obj)
        if oid in seen or getattr(obj, "_trace_internal", False):
            return
        seen.add(oid)
        if isinstance(obj, mx.array):
            found.setdefault(oid, path)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, f"{path}.{k}" if path else str(k))
            # nn.Module subclasses dict; its plain attributes live in __dict__
            if hasattr(obj, "__dict__"):
                for k, v in vars(obj).items():
                    visit(v, f"{path}.{k}" if path else str(k))
            return
        if isinstance(obj, (list, tuple)):
            # dot style so weight paths compose with scope tree paths
            for i, v in enumerate(obj):
                visit(v, f"{path}.{i}")
            return
        if isinstance(obj, set):
            for v in obj:
                visit(v, f"{path}{{}}")
            return
        if callable(obj):
            closure = getattr(obj, "__closure__", None)
            if closure:
                for i, cell in enumerate(closure):
                    visit(cell.cell_contents, f"{path}<cell{i}>")
            for k, v in getattr(obj, "__globals__", {}).items():
                if isinstance(v, (mx.array, list, tuple, dict)) and not isinstance(v, type):
                    visit(v, f"{path}<global:{k}>")
        if hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                visit(v, f"{path}.{k}" if path else str(k))

    visit(root, "")
    return found


def state_holders(root: object) -> list[tuple[str, object]]:
    """(dot path, object) for every plain Python object reachable from root
    that holds an array: a KV cache, a namespace of buffers. Modules,
    containers, and arrays are not holders. The model reaches such state
    only through the object's own methods, which is what makes those calls
    recordable and replayable as one unit."""
    found: list[tuple[str, object]] = []
    seen: set[int] = set()

    def visit(obj: object, path: str) -> bool:
        oid = id(obj)
        if oid in seen or getattr(obj, "_trace_internal", False):
            return False
        seen.add(oid)
        if isinstance(obj, mx.array):
            return True
        hit = False
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                hit |= visit(v, f"{path}.{k}" if path else str(k))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                hit |= visit(v, f"{path}.{i}")
        elif isinstance(obj, set) or callable(obj) or not hasattr(obj, "__dict__"):
            return False
        if hasattr(obj, "__dict__"):
            for k, v in list(vars(obj).items()):
                hit |= visit(v, f"{path}.{k}" if path else str(k))
        if hit and not isinstance(obj, (dict, list, tuple)):
            found.append((path, obj))
        return hit

    visit(root, "")
    return found


def flatten_arrays(tree: object) -> list[mx.array]:
    """Every array in a nested tree of lists, tuples, and dicts, in order."""
    out: list[mx.array] = []

    def walk(obj: object) -> None:
        if isinstance(obj, mx.array):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)

    walk(tree)
    return out
