"""Sandbox protocol: the JSON job spec and verdict, plus the
parent side that spawns one worker per evaluation.

Metal reads MTL_SHADER_VALIDATION and MTL_CAPTURE_ENABLED at process launch,
so a mode is an environment set at spawn, never toggled in-process.
Validation alone reports only to os_log, so validate
mode also sets MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 and the parent scans
child stderr for "Invalid device load"/"Invalid device store" lines.

Kill-and-relaunch is routine, not an error path: an infinite-loop kernel
truly hangs mx.eval (verified: no OS-side error within 30s), killing the
child is how the wedged GPU recovers, and the wall timeout maps to a
structured verdict rather than an exception.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

# Never configurable: a timed run past this multiple of the library region
# time fails the child-side watchdog. The gate exists to catch wedged kernels,
# not honest slowness (the ship margin owns speed), and correct starting
# kernels for fused-op chains sit near 10x, so the factor is 20.
WATCHDOG_FACTOR = 20.0

_METAL_ENV = (
    "MTL_SHADER_VALIDATION",
    "MTL_SHADER_VALIDATION_REPORT_TO_STDERR",
    "MTL_CAPTURE_ENABLED",
)

MODES = {
    "validate": {"MTL_SHADER_VALIDATION": "1", "MTL_SHADER_VALIDATION_REPORT_TO_STDERR": "1"},
    "score": {},
    "capture": {"MTL_CAPTURE_ENABLED": "1"},  # debug only, off the hot path
}

_VALIDATION_SIGNALS = ("Invalid device load", "Invalid device store")
_VALIDATION_LINE_CAP = 10
_TAIL_CHARS = 4000


@dataclass(frozen=True)
class EvalSetSpec:
    """One eval set as the child sees it: k input files and k reference files
    (safetensors keyed "a<array_id>"), plus the span at this size when the
    shapes differ from the primary span."""

    label: str
    inputs_paths: tuple[str, ...]
    reference_paths: tuple[str, ...]
    t_library_ms: float | None       # required for the first set; informational
    correctness_only: bool           # True routes the set to the sweep gate
    nodes_json: str | None


@dataclass(frozen=True)
class LadderSpec:
    """One kernel's trip up the ladder, as the child sees it. The child rebuilds
    everything from this JSON; tolerances cross only this harness-to-harness
    boundary and never reach the judge.

    phase "validate" runs gates 2-8 with shader validation on; phase "score"
    re-runs smoke and determinism on the clean pipeline and then the ship
    clock, because validation recompiles pipelines and the timed pipeline must
    be the checked pipeline."""

    kernel: dict                      # KernelSpec fields
    assoc_tag: str                    # "preserving" | "changing"
    nodes_json: str                   # the primary region span
    input_ids: tuple[int, ...]        # kernel input order = these ids' order
    output_ids: tuple[int, ...]
    eval_sets: tuple[EvalSetSpec, ...]
    tolerances: dict                  # {"rtol": float, "atol": float}
    kappa: float
    changing_floor: float | None      # None -> the child computes the k-set spread floor
    min_win_ms: float                 # absolute term of the ship margin, copies folded in
    phase: str                        # "validate" | "score"
    seed: int = 0
    clock_pairs: int = 32             # ABBA pairs behind the ship clock
    weight_inputs: tuple[bool, ...] = ()  # per input: a model weight, never perturbed
    kind: str = "ladder"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(text: str) -> "LadderSpec":
        d = json.loads(text)
        if d.get("kind") != "ladder":
            raise ValueError(f"not a ladder spec: kind={d.get('kind')!r}")
        d["input_ids"] = tuple(d["input_ids"])
        d["output_ids"] = tuple(d["output_ids"])
        d["weight_inputs"] = tuple(d.get("weight_inputs", ()))
        d["eval_sets"] = tuple(
            EvalSetSpec(
                label=e["label"],
                inputs_paths=tuple(e["inputs_paths"]),
                reference_paths=tuple(e["reference_paths"]),
                t_library_ms=e["t_library_ms"],
                correctness_only=e["correctness_only"],
                nodes_json=e["nodes_json"],
            )
            for e in d["eval_sets"]
        )
        return LadderSpec(**d)


@dataclass(frozen=True)
class Verdict:
    """One evaluation's outcome. detail is the failed gate's structured detail
    (flat; empty on a pass), plus a "validation" entry the parent merges in
    validate mode. timing carries ms figures and is never part of any
    verdict-equality check: the same spec re-run must agree on everything else.
    """

    passed: bool
    failed_gate: str | None           # the gate that failed, or "subprocess"
    gates_passed: tuple[str, ...]
    detail: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(text: str) -> "Verdict":
        d = json.loads(text)
        d["gates_passed"] = tuple(d["gates_passed"])
        if d.get("detail") is not None and not isinstance(d["detail"], dict):
            d["detail"] = {"raw": str(d["detail"])}  # a corrupt child cannot crash the parent
        return Verdict(**d)


def mode_env(mode: str) -> dict[str, str]:
    """The child environment for a mode: the parent's environment with every
    Metal knob removed, then the mode's own vars set. Exposed for tests."""
    if mode not in MODES:
        raise ValueError(f"unknown sandbox mode {mode!r}; modes are {sorted(MODES)}")
    env = {k: v for k, v in os.environ.items() if k not in _METAL_ENV}
    env.update(MODES[mode])
    return env


def run_job(spec: LadderSpec, mode: str, timeout_s: float) -> Verdict:
    """Spawn one worker, write the spec to its stdin, read the one JSON verdict
    line from its stdout. Nonzero exit, crash, or wall timeout maps to
    failed_gate "subprocess" with the stderr tail."""
    env = mode_env(mode)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "autotuner.sandbox.worker"],
            input=spec.to_json(),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        return _subprocess_verdict(
            f"wall timeout after {timeout_s}s; child killed", _tail(e.stderr)
        )
    except OSError as e:
        # spawn failure under memory pressure is an evaluation failure too
        return _subprocess_verdict(f"child spawn failed: {e}", "")
    if proc.returncode != 0:
        return _subprocess_verdict(f"child exit {proc.returncode}", _tail(proc.stderr))
    verdict = _parse_verdict(proc.stdout)
    if verdict is None:
        return _subprocess_verdict(
            "child exited 0 without a verdict line", _tail(proc.stderr)
        )
    if mode == "validate":
        _merge_validation(verdict, proc.stderr)
    return verdict


def _subprocess_verdict(reason: str, stderr_tail: str) -> Verdict:
    return Verdict(
        passed=False,
        failed_gate="subprocess",
        gates_passed=(),
        detail={"reason": reason, "stderr_tail": stderr_tail},
    )


def _tail(text: str | bytes | None) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return text[-_TAIL_CHARS:]


def _parse_verdict(stdout: str) -> Verdict | None:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return Verdict.from_json(lines[-1])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def _merge_validation(verdict: Verdict, stderr: str) -> None:
    """Validation reporting is asynchronous, so the check runs after the child
    exits: scan its stderr for the shader-validation lines and put
    the evidence in the verdict detail for the ladder to act on."""
    matched = [l for l in stderr.splitlines() if any(s in l for s in _VALIDATION_SIGNALS)]
    if not matched:
        return
    verdict.detail["validation"] = {
        "counts": {s: sum(s in l for l in matched) for s in _VALIDATION_SIGNALS},
        "lines": matched[:_VALIDATION_LINE_CAP],
    }
