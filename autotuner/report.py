"""report.json: everything a reader needs to trust or reproduce the job.

Per the spec's artifact block plus plan 5.12/5.13: per-region and
per-hypothesis rows, the step clocks before and after, the baseline choice,
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

from .log import SIGN_CONVENTION

_ENV_VARS = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "MTL_SHADER_VALIDATION",
             "MTL_SHADER_VALIDATION_REPORT_TO_STDERR", "MTL_CAPTURE_ENABLED")


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

    def add_region(self, *, fingerprint: str, ops: list[str], copies: int,
                   workloads: list[str], p: dict, t_orig_ms: dict, bound: str | None,
                   s_max: float | None, t_shipped_ms: dict | None = None,
                   close_rule: str | None = None) -> None:
        s = None
        if t_shipped_ms:
            s = {w: t_orig_ms[w] / t_shipped_ms[w] for w in t_shipped_ms if t_shipped_ms[w]}
        self.regions.append({
            "fingerprint": fingerprint, "ops": ops, "copies": copies,
            "workloads": workloads, "p": p, "T_orig_ms": t_orig_ms,
            "bound": bound, "s_max": s_max, "T_shipped_ms": t_shipped_ms,
            "s": s, "close_rule": close_rule,
        })

    def add_hypothesis(self, *, hypothesis_id: str, region: str, kind: str,
                       parent: str | None, verdict: str, failed_gate: str | None,
                       region_ms: float | None) -> None:
        self.hypotheses.append({
            "id": hypothesis_id, "region": region, "kind": kind, "parent": parent,
            "verdict": verdict, "failed_gate": failed_gate, "region_ms": region_ms,
        })

    def to_dict(self) -> dict:
        return {
            "sign_convention": SIGN_CONVENTION,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
            "stranded": self.stranded,
            "coverage": self.coverage,
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, default=str) + "\n")
