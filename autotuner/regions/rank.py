"""Ranking (spec "Pricing and ranking"): order by combined share, tie-break by
bound (fusion has the most to offer where the cost is traffic), overlap
bookkeeping for the loop."""

from __future__ import annotations

from .types import Region

_BOUND_ORDER = {"memory": 0, "launch": 1, "compute": 2}
REGION_FLOOR_P = 0.02
ROOFLINE_HAS_ROOM = 1.2


def apply_floor(regions: list[Region], floor: float = REGION_FLOOR_P,
                has_room: float = ROOFLINE_HAS_ROOM) -> list[Region]:
    """Drop candidates whose copies together fall under the floor, and regions
    where even the ideal kernel barely beats the library. Marks, not deletes:
    dropped candidates keep their named reason for the report."""
    kept = []
    for r in regions:
        if r.combined_p < floor:
            r.rejected = f"under floor: combined p {r.combined_p:.4f} < {floor}"
        elif r.roofline is not None and r.roofline.s_max < has_room:
            r.rejected = f"no headroom: s_max {r.roofline.s_max:.2f} < {has_room}"
        else:
            kept.append(r)
    return kept


def rank(regions: list[Region]) -> list[Region]:
    return sorted(
        regions,
        key=lambda r: (-r.combined_p,
                       _BOUND_ORDER.get(r.roofline.bound if r.roofline else "compute", 3)),
    )


def overlaps(a: Region, b: Region) -> bool:
    """Two regions overlap if any member spans share a node in any workload."""
    for ma in a.members:
        for mb in b.members:
            if ma.workload == mb.workload and \
               not (ma.end_seq < mb.start_seq or ma.start_seq > mb.end_seq):
                return True
    return False


def covered_by(candidate: Region, shipped: Region) -> bool:
    """Every member of candidate lies inside some member span of shipped, in
    the same workload: the bigger shipped cut already owns those ops."""
    for mc in candidate.members:
        inside = any(
            ms.workload == mc.workload
            and ms.start_seq <= mc.start_seq and mc.end_seq <= ms.end_seq
            for ms in shipped.members
        )
        if not inside:
            return False
    return True
