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


def free_members(candidate: Region, shipped: list[Region]) -> list:
    """The candidate's copies that touch no shipped cut. A copy inside a
    shipped span belongs to that kernel now, and a copy reaching past one
    cannot be judged either: its library clock was taken before the ship,
    so nothing says whether it beats the kernel already installed there."""
    return [
        mc for mc in candidate.members
        if not any(
            ms.workload == mc.workload
            and ms.start_seq <= mc.end_seq and mc.start_seq <= ms.end_seq
            for s in shipped for ms in s.members
        )
    ]


def covered_by(candidate: Region, shipped: Region) -> bool:
    """Every member of candidate touches some member span of shipped."""
    return not free_members(candidate, [shipped])
