"""Region data model (plan section 4)."""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class Region:
    fingerprint: str
    ops: tuple[str, ...]              # canonical op sequence
    members: list[Stretch] = field(default_factory=list)
    rejected: str | None = None       # named reason, or None if viable

    # pricing, filled by price.py per workload name; t_orig_ms sums every copy
    # (the region's share of the step), t_rep_ms is one representative copy
    # (what the roofline's one-copy floor compares against)
    t_orig_ms: dict[str, float] = field(default_factory=dict)
    t_rep_ms: dict[str, float] = field(default_factory=dict)
    p: dict[str, float] = field(default_factory=dict)
    p_rep: dict[str, float] = field(default_factory=dict)   # one copy's share
    stability: dict[str, float] = field(default_factory=dict)  # 0..1 per workload
    roofline: Roofline | None = None

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
