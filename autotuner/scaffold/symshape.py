"""Symbolic shapes for naive lowering (plan 5.9).

A Dim is one array dimension carrying its concrete value at every traced
instance plus two renderings: launch-grammar text for the host side and a
Metal int expression for the device side. Dims constant across instances
render as literals; varying dims render as inK.shape[j] accessors (grammar)
and inK_shape[j] reads (body, via the shape buffers mlx injects), so one
kernel is correct at every sweep size. A View is (shape, strides) over a
contiguous buffer; view ops are stride algebra.
"""

from __future__ import annotations

from dataclasses import dataclass


class NoScaffold(RuntimeError):
    """This op sequence has no naive lowering. reason is a stable slug the
    caller can log; the message adds the specifics."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class Dim:
    values: tuple[int, ...]  # concrete value per instance; index 0 is primary
    grammar: str             # launch-grammar text
    body: str                # Metal int expression text

    @property
    def is_literal(self) -> bool:
        return self.grammar == str(self.values[0])

    @property
    def is_one(self) -> bool:
        return all(v == 1 for v in self.values)


def lit(value: int, n_inst: int) -> Dim:
    return Dim((value,) * n_inst, str(value), str(value))


def accessor(k: int, axis: int, values: tuple[int, ...]) -> Dim:
    return Dim(values, f"in{k}.shape[{axis}]", f"in{k}_shape[{axis}]")


def dim_mul(a: Dim, b: Dim) -> Dim:
    if a.is_one:
        return b
    if b.is_one:
        return a
    values = tuple(x * y for x, y in zip(a.values, b.values))
    if a.is_literal and b.is_literal:
        return Dim(values, str(values[0]), str(values[0]))
    return Dim(values, f"({a.grammar} * {b.grammar})", f"({a.body} * {b.body})")


def prod_dims(dims: tuple[Dim, ...], n_inst: int) -> Dim:
    out = lit(1, n_inst)
    for d in dims:
        out = dim_mul(out, d)
    return out


def dims_equal(a: Dim, b: Dim) -> bool:
    return a.values == b.values


def shapes_equal(a: tuple[Dim, ...], b: tuple[Dim, ...]) -> bool:
    return len(a) == len(b) and all(dims_equal(x, y) for x, y in zip(a, b))


def broadcast_shapes(a: tuple[Dim, ...], b: tuple[Dim, ...]) -> tuple[Dim, ...]:
    """Numpy-style right-aligned broadcast over symbolic dims."""
    out: list[Dim] = []
    for i in range(1, max(len(a), len(b)) + 1):
        da = a[-i] if i <= len(a) else None
        db = b[-i] if i <= len(b) else None
        if da is None:
            out.append(db)
        elif db is None or db.is_one:
            out.append(da)
        elif da.is_one:
            out.append(db)
        elif dims_equal(da, db):
            out.append(da if da.is_literal else db if db.is_literal else da)
        else:
            raise NoScaffold(
                "broadcast-mismatch", f"dims {da.values} vs {db.values}"
            )
    return tuple(reversed(out))


@dataclass(frozen=True)
class View:
    shape: tuple[Dim, ...]
    strides: tuple[Dim, ...]


def contiguous_strides(shape: tuple[Dim, ...], n_inst: int) -> tuple[Dim, ...]:
    strides: list[Dim] = []
    acc = lit(1, n_inst)
    for d in reversed(shape):
        strides.append(acc)
        acc = dim_mul(acc, d)
    return tuple(reversed(strides))


def contiguous(shape: tuple[Dim, ...], n_inst: int) -> View:
    return View(shape, contiguous_strides(shape, n_inst))


def is_contiguous(view: View, n_inst: int) -> bool:
    want = contiguous_strides(view.shape, n_inst)
    for d, s, w in zip(view.shape, view.strides, want):
        if d.is_one:
            continue  # any stride works on a unit dim
        if s.values != w.values:
            return False
    return True


def permute(view: View, perm: list[int]) -> View:
    return View(
        tuple(view.shape[p] for p in perm),
        tuple(view.strides[p] for p in perm),
    )


def drop_axes(view: View, axes: set[int]) -> View:
    keep = [i for i in range(len(view.shape)) if i not in axes]
    return View(
        tuple(view.shape[i] for i in keep),
        tuple(view.strides[i] for i in keep),
    )


def insert_axis(view: View, axis: int, n_inst: int) -> View:
    shape = list(view.shape)
    strides = list(view.strides)
    shape.insert(axis, lit(1, n_inst))
    strides.insert(axis, lit(0, n_inst))
    return View(tuple(shape), tuple(strides))
