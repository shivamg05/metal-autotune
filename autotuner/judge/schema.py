"""The strict JSON boundary between harness and judge.

Everything the judge sends crosses validate_response, which rejects anything
malformed with a reason. The schema is also enforcement by omission: kernel
names are assigned by the harness, and init_value, math_mode, and streams are
not fields here, so the judge cannot set them; unknown keys are rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from autotuner_runtime.grammar import Expr, GrammarError
from autotuner_runtime.kernels import _DTYPES, KernelSpec, KernelStage

# common kinds, offered as suggestions; a kind is any short label the judge chooses
SUGGESTED_KINDS = ("on-chip", "specialize", "retile", "re-layout", "algorithm", "launch", "fix")
ASSOC_TAGS = ("preserving", "changing")
CONDITIONS = ("correct", "shipped", "failed")
# verdict outcomes the queue's conditions read
OUTCOMES = ("failed", "correct_slower", "tentative_ship", "shipped", "rolled_back")

_DTYPE_NAMES = frozenset(_DTYPES)  # one source of truth with the kernel call site
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_IN_REF = re.compile(r"in\d+\Z")
_SCRATCH = re.compile(r"tmp\d+\Z")
ITEM_ID_MAX_CHARS = 64  # ids become kernel filenames, with harness prefixes and suffixes
RESERVED_ITEM_IDS = frozenset({"scaffold", "scafix", "original", "head", "shipped"})


class MalformedResponse(ValueError):
    """The judge's JSON is outside the schema; the message says why."""


class JudgeBabble(RuntimeError):
    """Two malformed responses in a row for one call. The loop maps this to a
    failed hypothesis (it burns budget, so a babbling judge exhausts its
    region rather than stalling it); it never counts toward the 5-straight
    compile/static fail streak, which is about kernels on one parent."""


@dataclass(frozen=True)
class QueueItem:
    """One hypothesis: English, not Metal."""

    id: str
    kind: str                      # a short label for the move, in the judge's words
    assoc_tag: str                 # preserving | changing
    hypothesis: str                # English, not Metal
    family_id: str | None = None   # the judge's own grouping label; the harness reads nothing into it
    depends_on: str | None = None  # an earlier item's id
    condition: str | None = None   # correct | shipped | failed; paired with depends_on


@dataclass(frozen=True)
class KernelProposal:
    """One Metal edit of a named parent, for the front ready item. The harness
    owns names, boundary dtypes, init_value, math_mode, and streams.
    Stages declare their own intermediate shapes and dtypes."""

    source: str                                # kernel BODY
    parent_kernel_id: str                      # which code this change edits
    grid: tuple[str, str, str]                 # launch grammar, TOTAL threads
    threadgroup: tuple[str, str, str]
    output_shapes: tuple[tuple[str, ...], ...] # one per region output
    header: str = ""                           # empty means the parent's
    template: tuple[tuple[str, str], ...] = ()  # (name, dtype name or "inK"); empty means the parent's
    fallback_predicate: str | None = None       # true -> use the library path
    scratch: tuple[tuple[str, str, tuple[str, ...]], ...] = ()  # (tmpN, dtype, shape exprs)
    item_id: str | None = None                  # the queue item this kernel is for
    stages: tuple[KernelStage, ...] = ()
    target_workload: str | None = None          # workload whose speed this edit aims to improve


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
    lesson: str | None = None      # one sentence for later regions of this job


@dataclass(frozen=True)
class NextResponse:
    mutations: tuple[Mutation, ...]
    kernel: KernelProposal | None  # None yields; refused while the budget lasts
    lesson: str | None = None


def validate_response(obj: object) -> SeedResponse | NextResponse:
    """Validate one judge response. Seed shape is {"queue": [...]}; next shape
    is {"mutations": [...], "kernel": {...}|null}. Anything else is rejected."""
    if not isinstance(obj, dict):
        raise MalformedResponse(f"response must be a JSON object, got {type(obj).__name__}")
    lesson = _lesson(obj)
    keys = set(obj) - {"lesson"}
    if "queue" in keys:
        if keys != {"queue"}:
            raise MalformedResponse(f"a seed response has exactly the key 'queue', got {sorted(keys)}")
        return SeedResponse(queue=_seed_queue(obj["queue"]), lesson=lesson)
    if keys == {"mutations", "kernel"}:
        return NextResponse(
            mutations=_mutations(obj["mutations"]),
            kernel=None if obj["kernel"] is None else _proposal(obj["kernel"]),
            lesson=lesson,
        )
    raise MalformedResponse(
        f"response keys must be ['queue'] or ['kernel', 'mutations'], plus an optional "
        f"'lesson', got {sorted(keys)}"
    )


def _lesson(obj: dict) -> str | None:
    lesson = obj.get("lesson")
    if lesson is None:
        return None
    if not isinstance(lesson, str):
        raise MalformedResponse("lesson must be a string when present")
    # Prose cannot change execution. Bound its later prompt excerpt, rather
    # than discard an otherwise valid proposal because its note is long.
    return lesson.strip() or None


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
    if not _IDENT.match(item_id) or len(item_id) > ITEM_ID_MAX_CHARS:
        raise MalformedResponse(
            f"item id {item_id!r} must be letters, digits, and underscores, at most "
            f"{ITEM_ID_MAX_CHARS} characters: ids become kernel names")
    if item_id in RESERVED_ITEM_IDS:
        raise MalformedResponse(f"item id {item_id!r} is reserved for harness kernels or parent aliases")
    kind = " ".join(_str(obj, "kind", f"item {item_id!r}").split())
    if not kind:
        raise MalformedResponse(f"item {item_id!r}: kind must be a non-empty label")
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
                  "header", "template", "fallback_predicate", "scratch", "item_id", "stages",
                  "target_workload"}


def _proposal(obj: object) -> KernelProposal:
    if not isinstance(obj, dict):
        raise MalformedResponse(f"kernel must be an object or null, got {type(obj).__name__}")
    unknown = set(obj) - _PROPOSAL_KEYS
    if unknown:
        # init_value, math_mode, streams, and the kernel name land here by design
        raise MalformedResponse(f"kernel has unknown keys {sorted(unknown)}; they are not the judge's to set")
    stages = ()
    if "stages" in obj:
        if set(obj) & {"source", "grid", "threadgroup", "template", "scratch"}:
            raise MalformedResponse("stages replace top-level source/grid/threadgroup/template/scratch")
        if not isinstance(obj["stages"], list) or not obj["stages"]:
            raise MalformedResponse("stages must be a non-empty list of dispatches")
        stages = tuple(_stage(stage, i) for i, stage in enumerate(obj["stages"]))
    source = "// Ordered stages; see the stage sources.\n" if stages else _str(obj, "source", "kernel")
    parent = _str(obj, "parent_kernel_id", "kernel")
    grid = ("1", "1", "1") if stages else _exprs3(obj, "grid")
    threadgroup = ("1", "1", "1") if stages else _exprs3(obj, "threadgroup")
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
    target = _opt_str(obj, "target_workload", "kernel")
    if target is not None and not target.strip():
        raise MalformedResponse("kernel: 'target_workload' must be a non-empty workload name")
    return KernelProposal(source=source, parent_kernel_id=parent, grid=grid,
                          threadgroup=threadgroup, output_shapes=out_shapes,
                          header=header, template=template, fallback_predicate=fallback,
                          scratch=_scratch(obj.get("scratch", [])),
                          item_id=_opt_str(obj, "item_id", "kernel"), stages=stages,
                          target_workload=target)


def _stage(obj, index):
    keys = {"inputs", "outputs", "source", "header", "grid", "threadgroup",
            "output_shapes", "output_dtypes", "template"}
    if not isinstance(obj, dict) or set(obj) - keys:
        raise MalformedResponse(f"stage {index}: fields are {sorted(keys)}")
    for key, pattern in (("inputs", r"(?:in|tmp|out)\d+"), ("outputs", r"(?:tmp|out)\d+")):
        values = obj.get(key)
        if not isinstance(values, list) or any(not isinstance(v, str) or not re.fullmatch(pattern, v) for v in values):
            raise MalformedResponse(f"stage {index}: {key} must be a list of buffer names")
    dtypes = obj.get("output_dtypes")
    if not isinstance(dtypes, list) or any(not isinstance(dt, str) or dt not in _DTYPE_NAMES for dt in dtypes):
        raise MalformedResponse(f"stage {index}: output_dtypes must be known dtype names")
    # Reuse the existing source, shape, template and launch grammar validation.
    single = _proposal({k: v for k, v in obj.items() if k not in {"inputs", "outputs", "output_dtypes"}}
                       | {"parent_kernel_id": "stage"})
    if len(obj["outputs"]) != len(single.output_shapes) or len(dtypes) != len(obj["outputs"]):
        raise MalformedResponse(f"stage {index}: every output needs a shape and dtype")
    return KernelStage(tuple(obj["inputs"]), tuple(obj["outputs"]), KernelSpec(
        kernel_id=f"stage_{index}", name=f"stage_{index}",
        input_names=tuple(f"in{i}" for i in range(len(obj["inputs"]))),
        output_names=tuple(f"out{i}" for i in range(len(obj["outputs"]))),
        source=single.source, header=single.header, grid=single.grid,
        threadgroup=single.threadgroup, output_shapes=single.output_shapes,
        output_dtypes=tuple(dtypes), template=single.template))


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
