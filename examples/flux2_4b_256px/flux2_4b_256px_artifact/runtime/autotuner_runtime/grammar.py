"""The launch grammar: grid, threadgroup, output shapes, and
fallback predicates are expressions over the call's shape environment,
evaluated fresh at every call so one kernel launches correctly at every size.

Expressions are parsed with ast into a whitelisted node set; there is no eval.
Environment names: in0, in1, ... with .shape[i] and .ndim accessors. Functions:
min, max, ceil_div. Operators: + - * // % (and / as integer division),
comparisons, and/or/not. Values are ints and bools.
"""

from __future__ import annotations

import ast
from typing import Sequence


class GrammarError(ValueError):
    """The expression is outside the launch grammar; the message says where."""


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Div: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
}

_CMPOPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}

_FUNCS = {
    "min": min,
    "max": max,
    "ceil_div": lambda a, b: -(-a // b),
}


class Expr:
    """A parsed launch-grammar expression, reusable across calls."""

    def __init__(self, text: str):
        self.text = text
        try:
            self._tree = ast.parse(text, mode="eval").body
        except (SyntaxError, RecursionError, MemoryError) as e:
            # a 40k-term operator chain exhausts the parser's stack, not its
            # syntax rules; resource exhaustion is a grammar rejection too
            raise GrammarError(f"cannot parse {text!r}: {type(e).__name__}: {e}")
        _check(self._tree, text)

    def evaluate(self, shapes: Sequence[Sequence[int]]) -> int | bool:
        """shapes[i] is the concrete shape of input i for this call."""
        return _eval(self._tree, shapes, self.text)

    def __repr__(self) -> str:
        return f"Expr({self.text!r})"


def _check(node: ast.AST, text: str) -> None:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, bool)) or isinstance(node.value, float):
            raise GrammarError(f"{text!r}: only integer and boolean literals")
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        _check(node.left, text)
        _check(node.right, text)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.Not)):
        _check(node.operand, text)
        return
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or type(node.ops[0]) not in _CMPOPS:
            raise GrammarError(f"{text!r}: only single binary comparisons")
        _check(node.left, text)
        _check(node.comparators[0], text)
        return
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            _check(v, text)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise GrammarError(f"{text!r}: only min/max/ceil_div calls")
        if node.keywords or len(node.args) != 2:
            raise GrammarError(f"{text!r}: grammar calls take exactly two positional args")
        for a in node.args:
            _check(a, text)
        return
    if isinstance(node, ast.Attribute):
        if node.attr != "ndim":
            raise GrammarError(f"{text!r}: only .ndim and .shape[i] accessors")
        _input_index(node.value, text)
        return
    if isinstance(node, ast.Subscript):
        if not (isinstance(node.value, ast.Attribute) and node.value.attr == "shape"):
            raise GrammarError(f"{text!r}: only .shape[i] subscripts")
        _input_index(node.value.value, text)
        _check(node.slice, text)
        return
    raise GrammarError(f"{text!r}: {type(node).__name__} is outside the launch grammar")


def _input_index(node: ast.AST, text: str) -> int:
    if not (isinstance(node, ast.Name) and node.id.startswith("in") and node.id[2:].isdigit()):
        raise GrammarError(f"{text!r}: inputs are named in0, in1, ...")
    return int(node.id[2:])


def _eval(node: ast.AST, shapes: Sequence[Sequence[int]], text: str):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_eval(node.left, shapes, text), _eval(node.right, shapes, text))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, shapes, text)
        return -v if isinstance(node.op, ast.USub) else not v
    if isinstance(node, ast.Compare):
        return _CMPOPS[type(node.ops[0])](_eval(node.left, shapes, text), _eval(node.comparators[0], shapes, text))
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval(v, shapes, text) for v in node.values)
        return any(_eval(v, shapes, text) for v in node.values)
    if isinstance(node, ast.Call):
        return _FUNCS[node.func.id](*(_eval(a, shapes, text) for a in node.args))
    if isinstance(node, ast.Attribute):  # inN.ndim
        i = _input_index(node.value, text)
        _bounds(i, shapes, text)
        return len(shapes[i])
    if isinstance(node, ast.Subscript):  # inN.shape[i]
        i = _input_index(node.value.value, text)
        _bounds(i, shapes, text)
        axis = _eval(node.slice, shapes, text)
        try:
            return shapes[i][axis]
        except IndexError:
            raise GrammarError(f"{text!r}: axis {axis} out of range for input {i} with shape {tuple(shapes[i])}")
    raise GrammarError(f"{text!r}: unreachable node {type(node).__name__}")


def _bounds(i: int, shapes: Sequence[Sequence[int]], text: str) -> None:
    if i >= len(shapes):
        raise GrammarError(f"{text!r}: input {i} out of range, call has {len(shapes)} inputs")
