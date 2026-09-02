"""report.json: everything a reader needs to trust or reproduce the job.

It holds per-region and per-hypothesis rows, the step clocks before and after, the baseline choice,
every defaulted or tuned constant actually used, pinned versions, seeds, the
relevant environment variables, and the sign convention statement itself.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx


_ENV_VARS = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "MTL_SHADER_VALIDATION",
             "MTL_SHADER_VALIDATION_REPORT_TO_STDERR", "MTL_CAPTURE_ENABLED")

# what the short field names mean; every millisecond figure on a hypothesis
# is one pass of one copy of the region
LEGEND = {
    "p": "share of one step this region costs, all copies together, per workload",
    "T_orig_ms": "library time for all copies, per workload, from pricing",
    "T_rep_ms": "library time for one copy, per workload, from pricing",
    "roofline_ms": "the physical limit for one copy: bytes over bandwidth, or flops over peak",
    "s_max": "speedup ceiling: one copy's library time over its physical limit",
    "bound": "the resource that limits the region: memory, compute, or launch",
    "s": "speedup shipped: the library's time over the shipped kernel's, both from the kernel's own ship clock",
    "region_ms": "the kernel's time for one pass of one copy, from the ship clock",
    "library_ms": "the library's time for the same pass, measured beside it",
    "win_ms": "library_ms minus region_ms; positive means the kernel is faster",
    "sigma_ms": "uncertainty of win_ms; a ship needs a win past three of these",
    "stranded": "regions dropped before the search, grouped by reason",
}


@dataclass
class Report:
    manifest_path: str = ""
    constants: dict = field(default_factory=dict)     # every default/tunable actually used
    baseline: dict = field(default_factory=dict)      # plain vs compiled clocks + choice
    peaks: dict = field(default_factory=dict)
    session: dict = field(default_factory=dict)       # A/A floor, warmup stats
    step_ms: dict = field(default_factory=dict)       # workload -> before/after
    regions: list = field(default_factory=list)
    hypotheses: list = field(default_factory=list)
    stranded: list = field(default_factory=list)      # rejected regions with reasons
    coverage: dict = field(default_factory=dict)      # the standing self-proof line
    final: dict = field(default_factory=dict)         # the whole-model check after the last region

    def add_region(self, *, fingerprint: str, ops: list[str], copies: int,
                   workloads: list[str], p: dict, t_orig_ms: dict, bound: str | None,
                   s_max: float | None, t_shipped_ms: dict | None = None,
                   close_rule: str | None = None, t_rep_ms: dict | None = None,
                   roofline_ms: float | None = None, hypotheses: int = 0,
                   head_ms: float | None = None, speedup: float | None = None) -> None:
        s = speedup
        tally: dict[str, int] = {}
        for h in self.hypotheses:
            if h["region"] == fingerprint:
                tally[h["verdict"]] = tally.get(h["verdict"], 0) + 1
        self.regions.append({
            "fingerprint": fingerprint, "ops": ops, "copies": copies,
            "workloads": workloads, "p": p, "T_orig_ms": t_orig_ms,
            "T_rep_ms": t_rep_ms or {}, "roofline_ms": roofline_ms,
            "bound": bound, "s_max": s_max, "T_shipped_ms": t_shipped_ms,
            "s": s, "head_ms": head_ms, "hypotheses": hypotheses,
            "outcomes": tally, "close_rule": close_rule,
        })

    def add_hypothesis(self, *, hypothesis_id: str, region: str, kind: str,
                       parent: str | None, verdict: str, failed_gate: str | None,
                       region_ms: float | None, hypothesis_text: str = "",
                       assoc_tag: str | None = None, kernel: str | None = None,
                       library_ms: float | None = None, win_ms: float | None = None,
                       sigma_ms: float | None = None) -> None:
        self.hypotheses.append({
            "id": hypothesis_id, "region": region, "kind": kind,
            "hypothesis": hypothesis_text, "assoc_tag": assoc_tag, "parent": parent,
            "kernel": kernel, "verdict": verdict, "failed_gate": failed_gate,
            "region_ms": region_ms, "library_ms": library_ms, "win_ms": win_ms,
            "sigma_ms": sigma_ms,
        })

    def stranded_by_reason(self) -> list[dict]:
        groups: dict[str, list] = {}
        for row in self.stranded:
            groups.setdefault(row["reason"], []).append(
                {"fingerprint": row["fingerprint"], "ops": row.get("ops", [])})
        return [{"reason": reason, "count": len(rows), "regions": rows}
                for reason, rows in groups.items()]

    def to_dict(self) -> dict:
        return {
            "legend": LEGEND,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "machine": {
                "chip": mx.device_info().get("device_name", "unknown"),
                "memory_bytes": mx.device_info().get("memory_size"),
                "os": platform.platform(),
                "python": platform.python_version(),
                "mlx": mx.__version__,
            },
            "env": {k: os.environ.get(k) for k in _ENV_VARS},
            "manifest": self.manifest_path,
            "constants": self.constants,
            "baseline": self.baseline,
            "peaks": self.peaks,
            "session": self.session,
            "step_ms": self.step_ms,
            "regions": self.regions,
            "hypotheses": self.hypotheses,
            "stranded": self.stranded_by_reason(),
            "coverage": self.coverage,
            "final": self.final,
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, default=str) + "\n")
