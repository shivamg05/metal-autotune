"""One replay for every size of a dimension, from replays at a few sizes.

A replay wrapper is exact only at the size it was recorded at: reshape
targets, slice bounds and state-call arguments are literals from the trace.
Emit the same scope's replay at several recorded sizes and compare the
sources token by token. Every token must be identical except integers, and
each integer that differs must be one exact integer affine function of the
size (a * n + b, n read from an argument's shape) at every recorded size;
anything else is not generalized. The result reads the size from its
argument and computes those integers from it.

This establishes nothing about sizes that were not recorded: the model's
Python may branch on a size the recordings never crossed. A caller serves a
size only after checking the generalized replay against the original there.
"""

from __future__ import annotations

import io
import re
import tokenize
from dataclasses import dataclass

from .emit import EmittedWrapper, NotReplayable, _array_refs, emit_wrapper, scope_nodes


class NotSizeGeneric(ValueError):
    """The replays differ in a way no single size expression explains."""


def _tokens(source: str) -> list[tokenize.TokenInfo]:
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}
    return [t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type not in skip]


def _fit(values: list[int], sizes: list[int]) -> tuple[int, int] | None:
    """Integers a, b with a * n + b == value at every recorded size."""
    (n0, v0), (n1, v1) = (sizes[0], values[0]), (sizes[1], values[1])
    if (v1 - v0) % (n1 - n0):
        return None
    a = (v1 - v0) // (n1 - n0)
    b = v0 - a * n0
    return (a, b) if all(a * n + b == v for n, v in zip(sizes, values)) else None


def _render(a: int, b: int, name: str) -> str:
    term = name if a == 1 else f"{a} * {name}"
    if a == 0:
        return str(b)
    if b == 0:
        return term if a == 1 else f"({term})"
    return f"({term} {'+' if b > 0 else '-'} {abs(b)})"


def generalize(sources: list[str], sizes: list[int], name: str = "_n") -> tuple[str, int]:
    """One source for all the given ones: each integer that varies with the
    size becomes an expression of `name`. Returns the source and how many
    integers were generalized. Needs three sizes, so every fit is checked by
    a size it was not derived from."""
    if len(sources) != len(sizes) or len(set(sizes)) < 3:
        raise NotSizeGeneric("generalizing needs replays at three or more distinct sizes")
    streams = [_tokens(s) for s in sources]
    if len({len(s) for s in streams}) != 1:
        raise NotSizeGeneric("the replays run different operations at different sizes")
    out, varied = [], 0
    for column in zip(*streams):
        first = column[0]
        if all(t.string == first.string for t in column):
            out.append(first)
            continue
        if not all(t.type == tokenize.NUMBER and t.string.isdigit() for t in column):
            raise NotSizeGeneric(f"line {first.start[0]}: {first.string!r} changes with size and is not an integer")
        fit = _fit([int(t.string) for t in column], sizes)
        if fit is None:
            raise NotSizeGeneric(f"line {first.start[0]}: {[t.string for t in column]} is not one "
                                 f"integer linear function of the size {sizes}")
        out.append(first._replace(type=tokenize.NAME, string=_render(*fit, name)))
        varied += 1
    return _untokenize(out), varied


def _untokenize(tokens: list[tokenize.TokenInfo]) -> str:
    """Rebuild source from the first replay's layout: each token keeps its line
    and column, and a replaced integer shifts the rest of its line."""
    lines: dict[int, list[str]] = {}
    shift: dict[int, int] = {}
    for t in tokens:
        if t.type in (tokenize.NEWLINE, tokenize.ENDMARKER, tokenize.INDENT, tokenize.DEDENT):
            continue
        row, col = t.start
        line = lines.setdefault(row, [])
        text = "".join(line)
        pad = col + shift.get(row, 0) - len(text)
        line.append(" " * max(pad, 0) + t.string)
        if t.end[0] == row:
            shift[row] = shift.get(row, 0) + len(t.string) - (t.end[1] - col)
    return "\n".join("".join(lines.get(r, [])) for r in range(1, max(lines) + 1)) + "\n"


# -- one scope's size-generic wrapper -----------------------------------------

SIZE_NAME = "_n"
_KERNEL_CALL = re.compile(
    r"^(?P<indent> *)_s = self\._specs\[(?P<kid>'[^']*')\]\n(?P=indent)_ins = (?P<ins>[^\n]*)\n"
    r"(?P=indent)_outs = _kernels\.try_call\(_s, _ins\)$", re.M)


@dataclass
class SizedWrapper:
    """A scope's replay with its size read from one argument dimension. The
    scope serves sizes lo..hi and each kernel its own range inside that; both
    are placeholders until the checks set them (finalize)."""

    scope_path: str
    template: str                     # class source with range placeholders
    kernel_ids: list[str]
    size_arg: str                     # the argument expression the size is read from
    size_dim: int
    sizes: list[int]                  # the size at each recording, primary first
    generalized: int                  # integers that became size expressions

    def finalize(self, class_name: str, scope_range: tuple[int, int],
                 kernel_ranges: dict[str, tuple[int, int] | None]) -> EmittedWrapper:
        """The shipped class: served sizes written into the code, a kernel
        with no range left on its original operations."""
        lo, hi = scope_range
        source = self.template.replace("class Layer(", f"class {class_name}(", 1)
        source = source.replace("__SCOPE_LO__", str(lo)).replace("__SCOPE_HI__", str(hi))

        def call(match):
            kid = match["kid"].strip("'")
            span = kernel_ranges.get(kid)
            indent = match["indent"]
            if span is None:
                return f"{indent}_outs = None  # {kid}: not checked at other sizes"
            return (f"{indent}_s = self._specs[{match['kid']}]\n{indent}_ins = {match['ins']}\n"
                    f"{indent}_outs = _kernels.try_sized(_s, _ins) if {span[0]} <= {SIZE_NAME} <= {span[1]} else None")

        source = _KERNEL_CALL.sub(call, source)
        served = [k for k in self.kernel_ids if kernel_ranges.get(k) is not None]
        return EmittedWrapper(class_name=class_name, scope_path=self.scope_path, source=source,
                              span_map=[], kernel_ids=served)


def _size_argument(variants) -> tuple[str, list, int]:
    """The first array argument, in call order, whose shape changes across
    the recordings: its access expression, enclosing containers and dim."""
    trace, scope, _ = variants[0]
    arguments = [(f"a{i}", t) for i, t in enumerate(scope.args_template)] + list(scope.kwargs_template.items())
    shapes = []
    for trace, scope, _ in variants:
        nodes = scope_nodes(trace, scope)
        specs = trace.span_specs(nodes[0].seq, nodes[-1].seq)
        shapes.append({expr: specs[scope.arg_ids[ref.index]][0]
                       for name, t in arguments for ref, expr, _ in _array_refs(t, name)
                       if scope.arg_ids[ref.index] in specs})
    for name, t in arguments:
        for ref, expr, containers in _array_refs(t, name):
            seen = [s.get(expr) for s in shapes]
            if None in seen or len({len(s) for s in seen}) != 1:
                continue
            for d in range(len(seen[0])):
                if len({s[d] for s in seen}) > 1:
                    return expr, containers, d
    raise NotSizeGeneric("no argument's shape changes with the size")


def _size_read(expr: str, containers, dim: int) -> str:
    """Read the size, or -1 when the call is not shaped like the recordings."""
    tests = []
    for base, kind, size in containers:
        tests.append(f"isinstance({base}, dict) and tuple({base}) == {size!r}" if kind == "dict"
                     else f"isinstance({base}, {kind}) and len({base}) == {size}")
    tests += [f"isinstance({expr}, mx.array)", f"{expr}.ndim > {dim}"]
    return f"{SIZE_NAME} = {expr}.shape[{dim}] if {' and '.join(tests)} else -1"


def sized_wrapper(variants: list[tuple], scope_path: str) -> SizedWrapper:
    """variants: (trace, scope call, splices) for the same call recorded at
    three or more sizes, primary first. Raises NotSizeGeneric or
    NotReplayable when one replay cannot serve every size between them."""
    expr, containers, dim = _size_argument(variants)
    sizes = []
    for trace, scope, _ in variants:
        nodes = scope_nodes(trace, scope)
        specs = trace.span_specs(nodes[0].seq, nodes[-1].seq)
        ref = next(r for n, t in [(f"a{i}", t) for i, t in enumerate(scope.args_template)]
                   + list(scope.kwargs_template.items()) for r, e, _ in _array_refs(t, n) if e == expr)
        sizes.append(specs[scope.arg_ids[ref.index]][0][dim])
    sources = [emit_wrapper(trace, scope, sorted(splices, key=lambda s: s.start_seq), "Layer").source
               for trace, scope, splices in variants]
    source, generalized = generalize(sources, sizes)
    header = re.search(r"^    def __call__\(self[^\n]*\n", source, re.M)
    if header is None or "        if not (" not in source:
        raise NotReplayable("the replay has no argument guard to read a size from")
    source = (source[:header.end()] + f"        {_size_read(expr, containers, dim)}\n"
              + source[header.end():])
    source = source.replace("        if not (", "        if not (__SCOPE_LO__ <= _n <= __SCOPE_HI__ and ", 1)
    kernel_ids = list(dict.fromkeys(s.kernel.kernel_id for s in variants[0][2]))
    return SizedWrapper(scope_path, source, kernel_ids, expr, dim, sizes, generalized)
