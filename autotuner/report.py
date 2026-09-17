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

from .log import json_safe


_ENV_VARS = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "MTL_SHADER_VALIDATION",
             "MTL_SHADER_VALIDATION_REPORT_TO_STDERR", "MTL_CAPTURE_ENABLED")

# what the short field names mean; every millisecond figure on a hypothesis
# is one pass of one copy of the region
LEGEND = {
    "target_workload": "the workload nominated before timing; its whole-model win must repeat before installation",
    "timing_case": "the captured input case behind this hypothesis's local kernel timings",
    "timing_workload": "the workload behind a region's recorded shipped-kernel speedup",
    "model_ratios": "per-workload products of incremental model ratios; diagnostic estimates, not final measurements",
    "p": "share of one step this region costs, all copies together, per workload",
    "T_orig_ms": "library time for all copies, per workload, from pricing",
    "T_rep_ms": "library time for one copy, per workload, from pricing",
    "roofline_ms": "estimated ideal cost of one copy from boundary traffic and arithmetic",
    "s_max": "optimistic headroom estimate: library time divided by estimated ideal cost",
    "bound": "estimated limiting resource: memory, known compute, or launch; custom arithmetic may be unknown",
    "s": "speedup shipped: the library's time over the shipped kernel's, both from the kernel's own ship clock",
    "region_ms": "the kernel's time for one pass of one copy, from the ship clock",
    "library_ms": "the library's time for the same pass, measured beside it",
    "win_ms": "library_ms minus region_ms; positive means the kernel is faster",
    "sigma_ms": "uncertainty of win_ms; a ship needs a win past three of these",
    "stranded": "regions dropped before the search, grouped by reason",
    "measured_compute_gflops": "the plain-matmul rate per dtype clocked in the region's own pricing window; "
                               "the compute term of its estimated limit divides the region's flops by it",
    "close_rule": "why a region's search ended: its budget or the job's is spent, the operator asked "
                  "to finish, or no discernible headroom (launch-bound, head unmoved by the last two "
                  "attempts, the last three kernels each within one sigma of the launch floor "
                  "clocked beside them)",
    "constants.openers_per_region": "how many attempts a region's widening round holds at most, each "
                                    "an opener written against the scaffold from a different "
                                    "direction; fewer when fewer directions can pay under the bound",
    "delivery": "how a kernel installs at the region's scope, per workload: direct, graph (the scope compiled with the cut inserted) or replay",
    "library_arm": "what the region's clocks ran the library as, per workload: plain ops, or one compiled graph where graph delivery compiles the scope",
    "baseline.clocks_ms.compiled": "under library inference: the model with its outermost compilable scopes compiled and empty, the baseline unless the manifest asks for plain",
    "final.delivery": "under a plain baseline: the untouched model against the installed scopes compiled with no kernel, and that against the patched model",
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
    accepted: list = field(default_factory=list)       # completed model checks, even before region close
    pricing: list = field(default_factory=list)        # latest measured opportunity, including rejected cuts
    stranded: list = field(default_factory=list)      # rejected regions with reasons
    coverage: dict = field(default_factory=dict)      # the standing self-proof line
    final: dict = field(default_factory=dict)         # the whole-model check after the last region

    def add_region(self, *, fingerprint: str, ops: list[str], copies: int,
                   workloads: list[str], p: dict, t_orig_ms: dict, bound: str | None,
                   s_max: float | None, t_shipped_ms: dict | None = None,
                   close_rule: str | None = None, t_rep_ms: dict | None = None,
                   roofline_ms: float | None = None, hypotheses: int = 0,
                   head_ms: float | None = None, speedup: float | None = None,
                   timing_workload: str | None = None, delivery: dict | None = None,
                   library_arm: dict | None = None) -> None:
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
            "timing_workload": timing_workload,
            "delivery": delivery or {}, "library_arm": library_arm or {},
            "outcomes": tally, "close_rule": close_rule,
        })

    def add_hypothesis(self, *, hypothesis_id: str, region: str, kind: str,
                       parent: str | None, verdict: str, failed_gate: str | None,
                       region_ms: float | None, hypothesis_text: str = "",
                       assoc_tag: str | None = None, kernel: str | None = None,
                       library_ms: float | None = None, win_ms: float | None = None,
                       sigma_ms: float | None = None, floor_ms: float | None = None,
                       summary: str = "", target_workload: str | None = None,
                       timing_case: str | None = None) -> None:
        self.hypotheses.append({
            "id": hypothesis_id, "region": region, "kind": kind,
            "hypothesis": hypothesis_text, "assoc_tag": assoc_tag, "parent": parent,
            "kernel": kernel, "verdict": verdict, "failed_gate": failed_gate,
            "region_ms": region_ms, "library_ms": library_ms, "win_ms": win_ms,
            "sigma_ms": sigma_ms, "floor_ms": floor_ms, "summary": summary,
            "target_workload": target_workload, "timing_case": timing_case,
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
            "accepted": self.accepted,
            "pricing": self.pricing,
            "stranded": self.stranded_by_reason(),
            "coverage": self.coverage,
            "final": self.final,
        }

    def write(self, path: str | Path) -> None:
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(json_safe(self.to_dict()), indent=1,
                                        default=str, allow_nan=False) + "\n")
        temporary.replace(path)
