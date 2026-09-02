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

import mlx.nn as nn


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
