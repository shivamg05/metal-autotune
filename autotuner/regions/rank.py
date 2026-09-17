"""Choose which overlapping alternatives to price, then rank measured headroom.

Discovery still keeps every legal cut. A CPU estimate only chooses the first
non-overlapping wave; deferred alternatives remain available if a selected cut
fails or cannot ship. Paired GPU prices, never the estimate, decide whether a
region has enough share to spend search attempts on it, and how much of the
step an ideal kernel there could remove decides the order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from .types import Region, Stretch

if TYPE_CHECKING:
    from ..measure.peaks import Peaks
    from ..trace.types import Trace

_BOUND_ORDER = {"memory": 0, "launch": 1, "compute": 2}
REGION_FLOOR_P = 0.02


def apply_floor(regions: list[Region], floor: float = REGION_FLOOR_P) -> list[Region]:
    """Drop candidates whose copies together fall under the floor. Marks, not
    deletes: dropped candidates keep their named reason for the report. Little
    headroom is not a reason to drop: rank() puts such regions last."""
    kept = []
    for r in regions:
        if r.rejected:
            continue
        if r.combined_p < floor:
            r.rejected = f"under floor: combined p {r.combined_p:.4f} < {floor}"
        else:
            kept.append(r)
    return kept


def rank(regions: list[Region]) -> list[Region]:
    """Prioritize possible time saved in the whole step, not region size alone.

    Amdahl gives step speedup 1 / (1 - p + p/s), so ranking the denominator's
    reduction p*(1-1/s) gives the same order without pretending the physical
    estimate is a speedup we have achieved. For arbitrary Metal, the boundary
    probe contributes a generic estimate while its unknown arithmetic is omitted.
    Sum reductions across workloads,
    matching the existing equal-weight combined-share convention.
    """
    return sorted(
        regions,
        key=lambda r: (-r.combined_removable_p, -r.combined_p,
                       _BOUND_ORDER.get(r.roofline.bound if r.roofline else "compute", 3),
                       r.fingerprint),
    )


@dataclass(frozen=True)
class RegionEstimate:
    """CPU-only scheduling hint. Neither field is a measured latency share."""

    combined_p: float
    removable_p: float


def estimate_regions(
    regions: list[Region],
    traces: Mapping[str, Trace],
    peaks: Peaks,
    step_ms: Mapping[str, float],
) -> dict[str, RegionEstimate]:
    """Estimate fusion opportunity using only recorded shapes and chip peaks.

    Compare the sum of each op's ideal cost with one fused region's ideal
    cost. This favors deletable memory traffic and launches over merely large
    compute regions. Actual kernels can fall short of either estimate, and a
    compiled baseline may already fuse ops, so these numbers only order work.
    In particular, zero estimated savings never rejects a singleton.
    """
    from .build import is_view
    from .roofline import stretch_roofline

    costs = {}
    for workload, trace in traces.items():
        prefix = [0.0]
        for node in trace.nodes:
            if is_view(node):
                cost = 0.0
            else:
                atom = Stretch(workload, node.seq, node.seq,
                               tuple(dict.fromkeys(node.in_arrays)),
                               tuple(dict.fromkeys(node.out_arrays)), node.module_stack)
                cost = stretch_roofline(trace, atom, peaks, 0.0).t_roofline_ms
            prefix.append(prefix[-1] + cost)
        costs[workload] = prefix

    estimates = {}
    for region in regions:
        share = removable = 0.0
        for member in region.members:
            step = step_ms.get(member.workload, 0.0)
            if step <= 0.0:
                continue  # missing timing cannot justify rejecting a candidate
            prefix = costs[member.workload]
            cost = prefix[member.end_seq + 1] - prefix[member.start_seq]
            floor = stretch_roofline(traces[member.workload], member, peaks, 0.0).t_roofline_ms
            share += cost / step
            removable += max(cost - floor, 0.0) / step
        estimates[region.fingerprint] = RegionEstimate(share, removable)
    return estimates


def select_frontier(
    regions: list[Region], estimates: Mapping[str, RegionEstimate],
) -> tuple[list[Region], dict[str, list[str]]]:
    """Select a non-overlapping pricing wave and name every deferred conflict.

    There is no top-N cutoff. Every candidate is selected or deferred behind
    an overlapping selected cut. Calling again after those cuts close exposes
    the alternatives, including shorter chains and singletons. No input region
    is modified or rejected; the caller retains the deferred queue.
    """
    empty = RegionEstimate(0.0, 0.0)
    ordered = sorted(
        (region for region in regions if not region.rejected),
        key=lambda r: (-estimates.get(r.fingerprint, empty).removable_p,
                       -estimates.get(r.fingerprint, empty).combined_p,
                       len(r.ops), r.fingerprint),
    )
    ready, deferred = [], {}
    for region in ordered:
        conflicts = [chosen.fingerprint for chosen in ready if any(
            a.workload == b.workload
            and a.start_seq <= b.end_seq and b.start_seq <= a.end_seq
            for a in region.members for b in chosen.members
        )]
        if conflicts:
            deferred[region.fingerprint] = conflicts
        else:
            ready.append(region)
    return ready, deferred


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
