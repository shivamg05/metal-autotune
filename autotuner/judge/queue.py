"""Harness-side queue state (spec "Hypothesis queue and conditionals").

The queue and the verdict log are the judge's only memory. pop_ready walks
from the front and returns the first item whose depends_on condition the
verdict log satisfies; mutations arrive from the judge after each verdict.
FamilyBook keeps the per-family_id counters (the spec's 8-strike abandonment)
and the family-being-climbed state; the head and shipped bookmarks stay in
the loop, never here.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .schema import (
    CONDITIONS,
    OUTCOMES,
    DeleteItem,
    InsertItem,
    Mutation,
    QueueItem,
    ReorderItems,
)

# which verdict outcomes satisfy each depends_on condition; a rolled-back ship
# failed in the end, so the failure branch reads it as failed
_SATISFIES = {
    "correct": frozenset({"correct_slower", "tentative_ship", "shipped"}),
    "shipped": frozenset({"shipped"}),
    "failed": frozenset({"failed", "rolled_back"}),
}
assert set(_SATISFIES) == set(CONDITIONS)


class QueueError(ValueError):
    """A mutation or verdict is inconsistent with the queue's state."""


class Queue:
    """One region's hypothesis queue plus its verdict log."""

    def __init__(self) -> None:
        self._items: list[QueueItem] = []
        self._verdicts: dict[str, str] = {}  # executed item id -> latest outcome
        self._in_flight: str | None = None   # popped, no verdict yet; its id stays taken

    def __len__(self) -> int:
        return len(self._items)

    @property
    def empty(self) -> bool:
        return not self._items

    @property
    def verdicts(self) -> dict[str, str]:
        return dict(self._verdicts)

    def ids(self) -> tuple[str, ...]:
        return tuple(it.id for it in self._items)

    def seed(self, items: Sequence[QueueItem]) -> None:
        """Load the judge's seed queue. Once per region, before any verdict."""
        if self._items or self._verdicts:
            raise QueueError("seed is only legal on an empty queue with no verdicts")
        seen: set[str] = set()
        for it in items:
            self._check_new_item(it, seen)
            seen.add(it.id)
        self._items = list(items)

    def pop_ready(self) -> QueueItem | None:
        """The front item whose depends_on condition holds, removed from the
        queue. Unsatisfied items are skipped, not consumed."""
        for i, item in enumerate(self._items):
            if self._satisfied(item):
                self._in_flight = item.id
                return self._items.pop(i)
        return None

    def record_verdict(self, item_id: str, outcome: str) -> None:
        """Log an executed item's outcome. Re-recording is legal: a rollback
        overwrites the ship it undoes."""
        if outcome not in OUTCOMES:
            raise QueueError(f"outcome {outcome!r} must be one of {list(OUTCOMES)}")
        if any(it.id == item_id for it in self._items):
            raise QueueError(f"item {item_id!r} is still queued; only executed items get verdicts")
        self._verdicts[item_id] = outcome
        if self._in_flight == item_id:
            self._in_flight = None

    def apply_mutations(self, mutations: Iterable[Mutation]) -> None:
        """Insert, delete, reorder, applied in order; the first bad mutation
        raises QueueError and leaves later ones unapplied."""
        for m in mutations:
            if isinstance(m, InsertItem):
                self._insert(m)
            elif isinstance(m, DeleteItem):
                self._delete(m)
            elif isinstance(m, ReorderItems):
                self._reorder(m)
            else:
                raise QueueError(f"unknown mutation {m!r}")

    def snapshot(self) -> tuple[dict, ...]:
        """The queue as the prompt renders it: items with satisfied flags."""
        return tuple(
            {
                "id": it.id,
                "kind": it.kind,
                "assoc_tag": it.assoc_tag,
                "family_id": it.family_id,
                "hypothesis": it.hypothesis,
                "depends_on": it.depends_on,
                "condition": it.condition,
                "satisfied": self._satisfied(it),
            }
            for it in self._items
        )

    def _satisfied(self, item: QueueItem) -> bool:
        if item.depends_on is None:
            return True
        outcome = self._verdicts.get(item.depends_on)
        return outcome is not None and outcome in _SATISFIES[item.condition]

    def _check_new_item(self, item: QueueItem, earlier: set[str]) -> None:
        if item.id in ("scaffold", "scafix"):
            # these tag harness-built kernels; a judge item using one would
            # silently overwrite the real kernel's identity
            raise QueueError(f"item id {item.id!r} is reserved for harness kernels")
        known = earlier | {it.id for it in self._items} | set(self._verdicts)
        if self._in_flight is not None:
            known.add(self._in_flight)
        if item.id in known:
            raise QueueError(f"item id {item.id!r} already exists in this region")
        if item.depends_on is not None and item.depends_on not in known:
            raise QueueError(
                f"item {item.id!r} depends on {item.depends_on!r}, which does not exist yet"
            )

    def _insert(self, m: InsertItem) -> None:
        self._check_new_item(m.item, set())
        if m.before is None:
            self._items.append(m.item)
            return
        for i, it in enumerate(self._items):
            if it.id == m.before:
                self._items.insert(i, m.item)
                return
        raise QueueError(f"insert before {m.before!r}: no such queued item")

    def _delete(self, m: DeleteItem) -> None:
        for i, it in enumerate(self._items):
            if it.id == m.item_id:
                stranded = [d.id for d in self._items if d.depends_on == m.item_id]
                if stranded:
                    raise QueueError(
                        f"delete {m.item_id!r} would strand {stranded}, which depend on it")
                del self._items[i]
                return
        raise QueueError(f"delete {m.item_id!r}: no such queued item")

    def _reorder(self, m: ReorderItems) -> None:
        current = {it.id: it for it in self._items}
        if sorted(m.order) != sorted(current):
            raise QueueError(
                f"reorder must permute exactly the queued ids {sorted(current)}, got {list(m.order)}"
            )
        self._items = [current[i] for i in m.order]


ABANDON_STRIKES = 8  # spec-fixed: correct-but-slower without ever beating the library


class FamilyBook:
    """Per-family_id counters for one region. The judge declares families; the
    harness counts. Beating the library (a region-clock ship) permanently
    clears a family from abandonment and resets its strikes."""

    def __init__(self) -> None:
        self._kernel_family: dict[str, str | None] = {}  # kernel id -> family (scaffold: None)
        self._strikes: dict[str, int] = {}
        self._beaten: set[str] = set()
        self._abandoned: set[str] = set()
        self._fresh = 0
        self.climbing: str | None = None  # family of the last correct-but-slower head update

    def register_scaffold(self, kernel_id: str) -> None:
        self._kernel_family[kernel_id] = None

    def register_kernel(self, kernel_id: str, family_id: str) -> None:
        self._kernel_family[kernel_id] = family_id

    def resolve(self, item: QueueItem, parent_kernel_id: str) -> str:
        """The family an executed item belongs to: its own declaration, else
        its parent kernel's family, else a new family (scaffold parent)."""
        family = item.family_id or self._kernel_family.get(parent_kernel_id)
        if family is None:
            self._fresh += 1
            family = f"family{self._fresh}"
        self._strikes.setdefault(family, 0)
        return family

    def record_verdict(self, family_id: str, outcome: str) -> None:
        if outcome not in OUTCOMES:
            raise QueueError(f"outcome {outcome!r} must be one of {list(OUTCOMES)}")
        self._strikes.setdefault(family_id, 0)
        if outcome == "correct_slower":
            self._strikes[family_id] += 1
            self.climbing = family_id
        elif outcome in ("tentative_ship", "shipped"):
            self._strikes[family_id] = 0
            self._beaten.add(family_id)
            if self.climbing == family_id:
                self.climbing = None
        # failed and rolled_back change no counter: the rolled-back ship
        # already marked the family beaten when it won the region clock

    def tripped(self, family_id: str) -> bool:
        """True on the 8th correct-but-slower of a family that never beat the
        library; the loop then abandons it and resets head."""
        return (family_id not in self._beaten
                and self._strikes.get(family_id, 0) >= ABANDON_STRIKES)

    def abandon(self, family_id: str) -> None:
        self._abandoned.add(family_id)
        if self.climbing == family_id:
            self.climbing = None

    def abandoned(self, family_id: str) -> bool:
        return family_id in self._abandoned

    def state(self) -> dict[str, dict]:
        """Per-family climb state and strike count, as the prompt renders it."""
        return {
            family: {
                "strikes": strikes,
                "beaten_library": family in self._beaten,
                "abandoned": family in self._abandoned,
            }
            for family, strikes in sorted(self._strikes.items())
        }
