"""The strict JSON boundary between harness and judge (plan 5.11, section 10).

Everything the judge sends crosses validate_response, which rejects anything
malformed with a reason. The schema is also enforcement by omission: kernel
names are assigned by the harness, and init_value, math_mode, and streams are
not fields here, so the judge cannot set them; unknown keys are rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from autotuner_runtime.grammar import Expr, GrammarError
from autotuner_runtime.kernels import _DTYPES

# the spec's menu; every hypothesis kind is one of these
KINDS = ("on-chip", "specialize", "retile", "re-layout", "algorithm", "launch", "fix")
ASSOC_TAGS = ("preserving", "changing")
CONDITIONS = ("correct", "shipped", "failed")
# verdict outcomes the queue's conditions read (plan section 4)
OUTCOMES = ("failed", "correct_slower", "tentative_ship", "shipped", "rolled_back")

_DTYPE_NAMES = frozenset(_DTYPES)  # one source of truth with the kernel call site
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_IN_REF = re.compile(r"in\d+\Z")
_SCRATCH = re.compile(r"tmp\d+\Z")


class MalformedResponse(ValueError):
    """The judge's JSON is outside the schema; the message says why."""


class JudgeBabble(RuntimeError):
    """Two malformed responses in a row for one call. The loop maps this to a
    failed hypothesis (it burns budget, so a babbling judge exhausts its
    region rather than stalling it); it never counts toward the 5-straight
    compile/static fail streak, which is about kernels on one parent."""


@dataclass(frozen=True)
class QueueItem:
    """One hypothesis: English, not Metal. Family rule: an item without
    family_id inherits its parent kernel's family at execution time, and an
    item whose parent has no family (the scaffold) starts a new one."""

    id: str
    kind: str                      # one of KINDS; carries the assoc tag
    assoc_tag: str                 # preserving | changing
    hypothesis: str                # English, not Metal
    family_id: str | None = None
    depends_on: str | None = None  # an earlier item's id
    condition: str | None = None   # correct | shipped | failed; paired with depends_on


@dataclass(frozen=True)
class KernelProposal:
    """One Metal edit of a named parent, for the front ready item. The harness
    owns the call site: no name, dtypes, init_value, math_mode, or streams."""

    source: str                                # kernel BODY
    parent_kernel_id: str                      # which code this change edits
    grid: tuple[str, str, str]                 # launch grammar, TOTAL threads
    threadgroup: tuple[str, str, str]
    output_shapes: tuple[tuple[str, ...], ...] # one per region output
    header: str = ""                           # empty means the parent's
    template: tuple[tuple[str, str], ...] = ()  # (name, dtype name or "inK"); empty means the parent's
    fallback_predicate: str | None = None       # true -> use the library path
    scratch: tuple[tuple[str, str, tuple[str, ...]], ...] = ()  # (tmpN, dtype, shape exprs)


@dataclass(frozen=True)
class InsertItem:
    item: QueueItem
    before: str | None = None  # queue id to insert before; None appends


@dataclass(frozen=True)
class DeleteItem:
    item_id: str


@dataclass(frozen=True)
class ReorderItems:
    order: tuple[str, ...]  # exact permutation of the current queue ids


Mutation = InsertItem | DeleteItem | ReorderItems


@dataclass(frozen=True)
class SeedResponse:
    queue: tuple[QueueItem, ...]


@dataclass(frozen=True)
class NextResponse:
    mutations: tuple[Mutation, ...]
    kernel: KernelProposal | None  # None yields: nothing left to propose


def validate_response(obj: object) -> SeedResponse | NextResponse:
    """Validate one judge response. Seed shape is {"queue": [...]}; next shape
    is {"mutations": [...], "kernel": {...}|null}. Anything else is rejected."""
    if not isinstance(obj, dict):
        raise MalformedResponse(f"response must be a JSON object, got {type(obj).__name__}")
    keys = set(obj)
    if "queue" in keys:
        if keys != {"queue"}:
            raise MalformedResponse(f"a seed response has exactly the key 'queue', got {sorted(keys)}")
        return SeedResponse(queue=_seed_queue(obj["queue"]))
    if keys == {"mutations", "kernel"}:
        return NextResponse(
            mutations=_mutations(obj["mutations"]),
            kernel=None if obj["kernel"] is None else _proposal(obj["kernel"]),
        )
    raise MalformedResponse(
        f"response keys must be ['queue'] or ['kernel', 'mutations'], got {sorted(keys)}"
    )


def _seed_queue(obj: object) -> tuple[QueueItem, ...]:
    if not isinstance(obj, list) or not obj:
        raise MalformedResponse("queue must be a non-empty list of items")
    items = tuple(_item(o) for o in obj)
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise MalformedResponse(f"duplicate queue item id {it.id!r}")
        if it.depends_on is not None and it.depends_on not in seen:
            raise MalformedResponse(
                f"item {it.id!r} depends on {it.depends_on!r}, which is not an earlier item"
            )
        seen.add(it.id)
    return items


_ITEM_KEYS = {"id", "kind", "assoc_tag", "hypothesis", "family_id", "depends_on", "condition"}


def _item(obj: object) -> QueueItem:
    if not isinstance(obj, dict):
        raise MalformedResponse(f"queue item must be an object, got {type(obj).__name__}")
    unknown = set(obj) - _ITEM_KEYS
    if unknown:
        raise MalformedResponse(f"queue item has unknown keys {sorted(unknown)}")
    item_id = _str(obj, "id", "queue item")
    if not _IDENT.match(item_id):
        raise MalformedResponse(
            f"item id {item_id!r} must be letters, digits, and underscores: ids become kernel names")
    kind = _str(obj, "kind", f"item {item_id!r}")
    if kind not in KINDS:
        raise MalformedResponse(f"item {item_id!r}: kind {kind!r} is not on the menu {list(KINDS)}")
    assoc = _str(obj, "assoc_tag", f"item {item_id!r}")
    if assoc not in ASSOC_TAGS:
        raise MalformedResponse(f"item {item_id!r}: assoc_tag {assoc!r} must be one of {list(ASSOC_TAGS)}")
    hypothesis = _str(obj, "hypothesis", f"item {item_id!r}")
    family = _opt_str(obj, "family_id", f"item {item_id!r}")
    depends = _opt_str(obj, "depends_on", f"item {item_id!r}")
    condition = _opt_str(obj, "condition", f"item {item_id!r}")
    if (depends is None) != (condition is None):
        raise MalformedResponse(f"item {item_id!r}: depends_on and condition come together")
    if condition is not None and condition not in CONDITIONS:
        raise MalformedResponse(
            f"item {item_id!r}: condition {condition!r} must be one of {list(CONDITIONS)}"
        )
    return QueueItem(id=item_id, kind=kind, assoc_tag=assoc, hypothesis=hypothesis,
                     family_id=family, depends_on=depends, condition=condition)


_PROPOSAL_KEYS = {"source", "parent_kernel_id", "grid", "threadgroup", "output_shapes",
                  "header", "template", "fallback_predicate", "scratch"}


def _proposal(obj: object) -> KernelProposal:
    if not isinstance(obj, dict):
        raise MalformedResponse(f"kernel must be an object or null, got {type(obj).__name__}")
    unknown = set(obj) - _PROPOSAL_KEYS
    if unknown:
        # init_value, math_mode, streams, and the kernel name land here by design
        raise MalformedResponse(f"kernel has unknown keys {sorted(unknown)}; they are not the judge's to set")
    source = _str(obj, "source", "kernel")
    parent = _str(obj, "parent_kernel_id", "kernel")
    grid = _exprs3(obj, "grid")
    threadgroup = _exprs3(obj, "threadgroup")
    shapes = obj.get("output_shapes")
    if not isinstance(shapes, list) or not shapes:
        raise MalformedResponse("kernel output_shapes must be a non-empty list, one expr list per output")
    out_shapes = tuple(
        tuple(_expr(e, f"output_shapes[{i}]") for e in _expr_list(s, f"output_shapes[{i}]"))
        for i, s in enumerate(shapes)
    )
    header = obj.get("header", "")
    if not isinstance(header, str):
        raise MalformedResponse("kernel header must be a string")
    template = _template(obj.get("template", []))
    fallback = obj.get("fallback_predicate")
    if fallback is not None:
        if not isinstance(fallback, str):
            raise MalformedResponse("fallback_predicate must be a string expression")
        _expr(fallback, "fallback_predicate")
    return KernelProposal(source=source, parent_kernel_id=parent, grid=grid,
                          threadgroup=threadgroup, output_shapes=out_shapes,
                          header=header, template=template, fallback_predicate=fallback,
                          scratch=_scratch(obj.get("scratch", [])))


def _scratch(obj: object) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Extra device buffers the kernel writes: [name, dtype, [shape exprs]]."""
    if not isinstance(obj, list):
        raise MalformedResponse("scratch must be a list of [name, dtype, [shape exprs]] entries")
    out = []
    for entry in obj:
        if not (isinstance(entry, list) and len(entry) == 3 and isinstance(entry[0], str)
                and isinstance(entry[1], str)):
            raise MalformedResponse(f"scratch entry {entry!r} must be [name, dtype, [shape exprs]]")
        name, dtype, shape = entry
        if not _SCRATCH.match(name):
            raise MalformedResponse(f"scratch buffer {name!r} must be named tmp0, tmp1, ...")
        if dtype not in _DTYPE_NAMES:
            raise MalformedResponse(f"scratch buffer {name!r}: unknown dtype {dtype!r}")
        exprs = tuple(_expr(e, f"scratch {name}") for e in _expr_list(shape, f"scratch {name} shape"))
        out.append((name, dtype, exprs))
    return tuple(out)


def _template(obj: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(obj, list):
        raise MalformedResponse("template must be a list of [name, value] pairs")
    pairs = []
    for pair in obj:
        if not (isinstance(pair, list) and len(pair) == 2
                and all(isinstance(x, str) for x in pair)):
            raise MalformedResponse(f"template entry {pair!r} must be a [name, value] string pair")
        name, value = pair
        if not _IDENT.match(name):
            raise MalformedResponse(f"template name {name!r} is not a C identifier")
        if value not in _DTYPE_NAMES and not _IN_REF.match(value):
            raise MalformedResponse(
                f"template value {value!r} must be a dtype name or an input reference like 'in0'"
            )
        pairs.append((name, value))
    return tuple(pairs)


def _mutations(obj: object) -> tuple[Mutation, ...]:
    if not isinstance(obj, list):
        raise MalformedResponse("mutations must be a list")
    return tuple(_mutation(m) for m in obj)


def _mutation(obj: object) -> Mutation:
    if not isinstance(obj, dict) or not isinstance(obj.get("op"), str):
        raise MalformedResponse(f"mutation must be an object with an 'op', got {obj!r}")
    op = obj["op"]
    if op == "insert":
        unknown = set(obj) - {"op", "item", "before"}
        if unknown:
            raise MalformedResponse(f"insert has unknown keys {sorted(unknown)}")
        before = _opt_str(obj, "before", "insert")
        return InsertItem(item=_item(obj.get("item")), before=before)
    if op == "delete":
        unknown = set(obj) - {"op", "id"}
        if unknown:
            raise MalformedResponse(f"delete has unknown keys {sorted(unknown)}")
        return DeleteItem(item_id=_str(obj, "id", "delete"))
    if op == "reorder":
        unknown = set(obj) - {"op", "order"}
        if unknown:
            raise MalformedResponse(f"reorder has unknown keys {sorted(unknown)}")
        order = obj.get("order")
        if not isinstance(order, list) or not all(isinstance(i, str) for i in order):
            raise MalformedResponse("reorder order must be a list of item ids")
        return ReorderItems(order=tuple(order))
    raise MalformedResponse(f"mutation op {op!r} must be insert, delete, or reorder")


def _str(obj: dict, key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise MalformedResponse(f"{where}: {key!r} must be a non-empty string, got {value!r}")
    return value


def _opt_str(obj: dict, key: str, where: str) -> str | None:
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise MalformedResponse(f"{where}: {key!r} must be a non-empty string when present")
    return value


def _expr_list(obj: object, where: str) -> list[str]:
    if not isinstance(obj, list) or not all(isinstance(e, str) for e in obj):
        raise MalformedResponse(f"kernel {where} must be a list of expression strings")
    return obj


def _exprs3(obj: dict, key: str) -> tuple[str, str, str]:
    value = _expr_list(obj.get(key), key)
    if len(value) != 3:
        raise MalformedResponse(f"kernel {key} must have exactly 3 expressions, got {len(value)}")
    return tuple(_expr(e, key) for e in value)


def _expr(text: str, where: str) -> str:
    try:
        Expr(text)
    except GrammarError as e:
        raise MalformedResponse(f"kernel {where}: {e}")
    return text
