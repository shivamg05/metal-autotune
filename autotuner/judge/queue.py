"""Harness-side queue state (spec "Hypothesis queue and conditionals").

The queue and the verdict log are the judge's only memory. pop_ready walks
from the front and returns the first item whose depends_on condition the
verdict log satisfies; mutations arrive from the judge after each verdict,
applied as one batch. The head and shipped bookmarks stay in the loop.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .schema import (
    CONDITIONS,
    OUTCOMES,
    RESERVED_ITEM_IDS,
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
        """Load the judge's seed queue: one batch of inserts, under the same
        rules as any later batch. Once per region, before any verdict."""
        if self._items or self._verdicts:
            raise QueueError("seed is only legal on an empty queue with no verdicts")
        self.apply_mutations([InsertItem(item=it) for it in items])

    def peek_ready(self) -> QueueItem | None:
        """The front item whose depends_on condition holds, left in place."""
        return next((item for item in self._items if self._satisfied(item)), None)

    def ready_item(self, wanted: str | None = None) -> QueueItem | None:
        """The front ready item, or the item named by wanted when it is
        ready, left in place."""
        ready = [item for item in self._items if self._satisfied(item)]
        if wanted is None:
            return ready[0] if ready else None
        return next((item for item in ready if item.id == wanted), None)

    def pop_ready(self, wanted: str | None = None) -> QueueItem | None:
        """Remove and return the front ready item, or the item named by
        wanted, which must itself be ready: a kernel written for one item is
        never evaluated under another's name. Unsatisfied items are skipped,
        not consumed."""
        chosen = self.ready_item(wanted)
        if chosen is None:
            return None
        self._items.remove(chosen)
        self._in_flight = chosen.id
        return chosen

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
        """Insert, delete, reorder, as one batch: all of it lands or none of
        it does, and the rules are judged against the queue the batch leaves
        behind. A dependency names an earlier item: one already run, or one
        queued ahead of the dependent, so deleting a parent together with its
        dependents is legal in either order, and no item can wait on itself
        or on one behind it."""
        items = list(self._items)
        for m in mutations:
            if isinstance(m, InsertItem):
                items = self._insert(items, m)
            elif isinstance(m, DeleteItem):
                items = self._delete(items, m)
            elif isinstance(m, ReorderItems):
                items = self._reorder(items, m)
            else:
                raise QueueError(f"unknown mutation {m!r}")
        earlier = self._executed()
        for it in items:
            if it.depends_on is not None and it.depends_on not in earlier:
                raise QueueError(
                    f"item {it.id!r} depends on {it.depends_on!r}, which is neither an item "
                    f"already run nor one queued ahead of it")
            earlier.add(it.id)
        self._items = items

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

    def _executed(self) -> set[str]:
        """Ids that have run, or are running: a dependency may name them, and
        no new item may reuse them."""
        return set(self._verdicts) | ({self._in_flight} - {None})

    def _insert(self, items: list[QueueItem], m: InsertItem) -> list[QueueItem]:
        new = m.item
        if new.id in RESERVED_ITEM_IDS:
            # Protect original starters and role aliases as well as scaffolds.
            raise QueueError(f"item id {new.id!r} is reserved for harness kernels or parent aliases")
        if new.id in self._executed() or any(it.id == new.id for it in items):
            raise QueueError(f"item id {new.id!r} already exists in this region")
        if m.before is None:
            return items + [m.item]
        for i, it in enumerate(items):
            if it.id == m.before:
                return items[:i] + [m.item] + items[i:]
        raise QueueError(f"insert before {m.before!r}: no such queued item")

    def _delete(self, items: list[QueueItem], m: DeleteItem) -> list[QueueItem]:
        kept = [it for it in items if it.id != m.item_id]
        if len(kept) == len(items):
            raise QueueError(f"delete {m.item_id!r}: no such queued item")
        return kept

    def _reorder(self, items: list[QueueItem], m: ReorderItems) -> list[QueueItem]:
        current = {it.id: it for it in items}
        if sorted(m.order) != sorted(current):
            raise QueueError(
                f"reorder must permute exactly the queued ids {sorted(current)}, got {list(m.order)}"
            )
        return [current[i] for i in m.order]

