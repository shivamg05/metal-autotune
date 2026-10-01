"""Module swap: resolve scope addresses on a live model, install and uninstall
wrapper instances.

Addresses are dot paths for serialization only; the swap always happens on
live instances by parent attribute assignment (or container index assignment
for children living in lists and dicts). Wrappers subclass nn.Module so the
wrapped subtree stays visible to parameters() and named_modules(), and their
__getattr__ raises AttributeError before 'wrapped' exists, which keeps
Module.__setattr__'s probe alive during __init__ (both platform-pinned).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def flatten_arrays(tree: object) -> list[mx.array]:
    """Every array in a nested tree of lists, tuples, and dicts, in order:
    how the recorder counts a call's outputs, and how a generated wrapper
    unpacks a replayed call that returns a structure.

    Walked with an explicit stack, not a recursive closure: on a state call's
    return (a KV cache's two views) the closure form measured 57 us/layer of
    GPU-side time that an iterative walk yielding the identical arrays does
    not, which was the whole cost of the attention wrapper."""
    out: list[mx.array] = []
    stack = [tree]
    while stack:
        obj = stack.pop()
        if isinstance(obj, mx.array):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            stack.extend(reversed(obj))
        elif isinstance(obj, dict):
            stack.extend(reversed(list(obj.values())))
    return out


def state_signature(value):
    """Structure, scalar settings and array shapes of state, never tensor data.

    Used to guard replays whose Python branches depended on a cache being
    empty or on its layout. Array contents remain live runtime inputs.
    """
    seen = {}

    def visit(obj):
        if isinstance(obj, mx.array):
            return ("array", tuple(obj.shape), str(obj.dtype))
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if id(obj) in seen:
            return ("alias", seen[id(obj)])
        seen[id(obj)] = len(seen)
        if isinstance(obj, dict):
            return ("dict", tuple((k, visit(v)) for k, v in sorted(obj.items())))
        if isinstance(obj, (tuple, list)):
            return (type(obj).__name__, tuple(visit(v) for v in obj))
        if hasattr(obj, "__dict__"):
            return (type(obj).__name__, visit(vars(obj)))
        raise TypeError(f"cannot describe state field of type {type(obj).__name__}")

    return visit(value)


class ReplayWrapper(nn.Module):
    """Base for generated replay wrappers. Subclasses implement __call__ as
    the scope's recorded op stream with shipped cuts spliced."""

    def __init__(self, wrapped: nn.Module, specs: dict | None = None):
        super().__init__()
        self.wrapped = wrapped
        self._specs = specs or {}

    def __getattr__(self, name):
        if name in self:
            return self[name]
        if "wrapped" not in self:
            raise AttributeError(name)
        return getattr(self["wrapped"], name)


def require_independent_models(first, second):
    """Weight arrays may be shared; mutable modules must belong to one arm."""
    if not isinstance(first, nn.Module) or not isinstance(second, nn.Module):
        return
    originals = {id(module) for module in first.modules()}
    for path, module in second.named_modules():
        if id(module) in originals:
            raise ValueError(
                f"build() reused a model module at {path or '<root>'}; return fresh model "
                "and layer instances on every call so installing a kernel cannot modify "
                "the untouched baseline. Weight arrays may be shared.")


def resolve(model: object, path: str):
    """path -> (container, key, child). Empty path means the model itself and
    has no parent; installing there is the caller's special case."""
    if path == "":
        raise ValueError("the root scope has no parent to swap under")
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        obj = _child(obj, part, path)
    last = parts[-1]
    return obj, last, _child(obj, last, path)


def resolve_value(model: object, path: str):
    """Resolve recorded weights/state through modules, dictionaries and sequences."""
    obj = model
    for part in path.split(".") if path else []:
        obj = _child(obj, part, path)
    return obj


def _child(obj: object, key: str, full_path: str):
    try:
        if isinstance(obj, (list, tuple)):
            return obj[int(key)]
        if isinstance(obj, dict) and not isinstance(obj, nn.Module):
            return obj[key]
        return getattr(obj, key)
    except (AttributeError, KeyError, IndexError, ValueError, TypeError) as e:
        raise KeyError(
            f"cannot resolve scope address {full_path!r}: no child {key!r} "
            f"on {type(obj).__name__} ({e}); the model tree changed since the trace"
        )


def install(model: object, path: str, wrapper: nn.Module) -> object:
    """Swap wrapper in at path; returns the previous occupant for rollback."""
    parent, key, occupant = resolve(model, path)
    _assign(parent, key, wrapper)
    return occupant


def uninstall(model: object, path: str, occupant: object) -> None:
    parent, key, _ = resolve(model, path)
    _assign(parent, key, occupant)


def _assign(parent: object, key: str, value: object) -> None:
    if isinstance(parent, list):
        parent[int(key)] = value
    elif isinstance(parent, dict) and not isinstance(parent, nn.Module):
        parent[key] = value
    elif isinstance(parent, tuple):
        raise TypeError("cannot swap a child living in a tuple; the model must use lists")
    else:
        setattr(parent, key, value)
