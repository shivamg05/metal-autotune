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
