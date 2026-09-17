"""Region data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .price import RegionPrice


@dataclass(frozen=True)
class Stretch:
    """One copy of a region in one workload's trace: a run of consecutive
    recorded calls plus its derived boundary."""

    workload: str
    start_seq: int
    end_seq: int                      # inclusive
    input_ids: tuple[int, ...]        # read from outside the stretch
    output_ids: tuple[int, ...]       # live outside: consumed later, or step outputs
    scope_stack: tuple[str, ...]      # common enclosing call chain of the member ops


@dataclass(frozen=True)
class Roofline:
    t_mem_ms: float
    t_compute_ms: float
    t_launch_ms: float
    t_roofline_ms: float
    bound: str                        # memory | compute | launch
    s_max: float
    t_floor_ms: float | None = None   # measured: one launch streaming the boundary, clocked beside the copy


@dataclass
class Region:
    fingerprint: str
    ops: tuple[str, ...]              # canonical op sequence
    members: list[Stretch] = field(default_factory=list)
    rejected: str | None = None       # named reason, or None if viable
    # how a kernel installs at the region's scope, per workload, read from the
    # record before any clock: direct (the kernel call alone), graph (the
    # scope compiled with the cut inserted), or replay (generated Python)
    delivery: dict[str, str] = field(default_factory=dict)
    delivery_reasons: dict[str, str] = field(default_factory=dict)   # why graph insertion declined

    def library_arm(self, workload: str, baseline: str) -> str:
        """What every clock runs the region's library ops as: one compiled
        graph where the deployed scope compiles, else as the baseline runs."""
        return "compiled" if baseline == "compiled" or self.delivery.get(workload) == "graph" else "plain"

    # pricing, filled by price.py per workload name; t_orig_ms sums every copy
    # (the region's share of the step), t_rep_ms is one representative copy
    # (what the roofline's one-copy floor compares against)
    t_orig_ms: dict[str, float] = field(default_factory=dict)
    t_rep_ms: dict[str, float] = field(default_factory=dict)
    t_floor_ms: dict[str, float] = field(default_factory=dict)   # one copy's probe, same window as t_rep_ms
    p: dict[str, float] = field(default_factory=dict)
    p_rep: dict[str, float] = field(default_factory=dict)   # one copy's share
    stability: dict[str, float] = field(default_factory=dict)  # 0..1 per workload
    roofline: Roofline | None = None
    rooflines: dict[str, Roofline] = field(default_factory=dict)  # each workload has its own measured floor
    prices: dict[str, RegionPrice] = field(default_factory=dict)  # one paired price per captured shape group

    @property
    def copies(self) -> int:
        return len(self.members)

    @property
    def workloads(self) -> tuple[str, ...]:
        seen: list[str] = []
        for m in self.members:
            if m.workload not in seen:
                seen.append(m.workload)
        return tuple(seen)

    @property
    def combined_p(self) -> float:
        return sum(self.p.values())

    @property
    def removable_p(self) -> dict[str, float]:
        """Amdahl's optimistic step reduction, p * (1 - 1/s_max).

        This is headroom, not a promised win. Each workload uses its own
        paired price and physical floor. ``roofline`` remains a fallback for
        older callers that supplied only one representative workload.
        """
        reductions = {}
        for workload, share in self.p.items():
            roof = self.rooflines.get(workload, self.roofline)
            reductions[workload] = (
                max(share, 0.0) * (1.0 - 1.0 / roof.s_max)
                if roof is not None and roof.s_max > 1.0 else 0.0
            )
        return reductions

    @property
    def combined_removable_p(self) -> float:
        """Sum of possible fractional savings, with equal workload weight.

        Like combined_p, this sum can exceed one across several workloads;
        it is a ranking score, never a claimed whole-job speedup.
        """
        return sum(self.removable_p.values())
