"""Sandbox protocol: the JSON job spec and verdict, plus the
parent side that spawns one worker per evaluation.

Metal reads MTL_SHADER_VALIDATION and MTL_CAPTURE_ENABLED at process launch,
so a mode is an environment set at spawn, never toggled in-process.
Validation alone reports only to os_log, so validate
mode also sets MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 and the parent scans
child stderr for "Invalid device load"/"Invalid device store" lines.

Workers share the desktop GPU. A short deadline surrounds each GPU evaluation,
separate from the whole-worker budget and cooling. After a timeout, recovery
must finish a checked GPU operation in a fresh worker, and see the GPU back near
its quiet speed, before search can continue.
"""

from __future__ import annotations

import dataclasses
import json
import os
import select
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

from autotuner.sandbox.watchdog import GPU_WINDOW_S
from autotuner.sandbox.recovery import QUIET_WAIT_S

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
    changing_floor: float | None      # legacy wire field; unused by current numeric policy
    min_win_ms: float                 # absolute term of the ship margin, copies folded in
    phase: str                        # "validate" | "score"
    seed: int = 0
    clock_pairs: int = 32             # ABBA pairs behind the ship clock
    weight_inputs: tuple[bool, ...] = ()  # per input: a model weight, never perturbed
    baseline: str = "plain"           # "compiled": time the library span as one compiled graph
    compute_floor_ms: float = 0.0     # the roofline's flops term, arithmetic; the probe covers the rest
    timing_incumbent: dict | None = None
    defer_cooling: bool = False       # only when the parent owns the remaining cooling deadline
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


_RECOVERY = [sys.executable, "-m", "autotuner.sandbox.recovery"]
_quiet_ms: float | None = None


def _quiet_reference() -> float:
    """The recovery matmul's time on a quiet GPU, taken once per process
    before any candidate runs, while no worker can have left work behind."""
    global _quiet_ms
    if _quiet_ms is None:
        probe = _supervise(_RECOVERY, "", mode_env("score"), GPU_WINDOW_S + 10.0)
        lines = probe.stdout.split() if isinstance(probe, subprocess.CompletedProcess) else []
        if len(lines) != 2 or lines[0] != "GPU_QUIET_MS" or probe.returncode != 0:
            raise WorkerFailed("could not record the GPU's quiet reference time: " +
                               (probe.detail["reason"] if isinstance(probe, Verdict)
                                else _tail(probe.stderr) or probe.stdout))
        _quiet_ms = float(lines[1])
    return _quiet_ms


def run_job(spec: LadderSpec, mode: str, timeout_s: float) -> Verdict:
    """Spawn one worker, write the spec to its stdin, read the one JSON verdict
    line from its stdout. Worker failures retain their stderr tail. A timeout
    becomes a candidate rejection only after the recovery check succeeds."""
    quiet_ms = _quiet_reference()
    started = time.monotonic()
    proc = _supervise(
        [sys.executable, "-m", "autotuner.sandbox.worker"],
        spec.to_json(), mode_env(mode), timeout_s,
    )
    if isinstance(proc, Verdict):
        proc.detail.update(kernel_id=spec.kernel.get("kernel_id"), phase=spec.phase)
        if proc.detail.get("failure_kind") == "timeout" and proc.detail.get("worker_exited"):
            return _recover_timeout(proc, started, quiet_ms, defer_cooling=spec.defer_cooling)
        return proc
    verdict = _parse_verdict(proc.stdout)
    if verdict is None:
        verdict = _subprocess_verdict(
            "child exited 0 without a verdict line", _tail(proc.stderr)
        )
        verdict.detail.update(kernel_id=spec.kernel.get("kernel_id"), phase=spec.phase)
        return verdict
    if mode == "validate":
        _merge_validation(verdict, proc.stderr)
    return verdict


def _recover_timeout(failure: Verdict, started: float, quiet_ms: float, *,
                     defer_cooling: bool) -> Verdict:
    """A timed-out candidate stays rejected; search resumes only once a trusted
    GPU check is correct and the GPU is back near its quiet speed."""
    probe_started = time.monotonic()
    probe = _supervise(_RECOVERY, json.dumps({"quiet_ms": quiet_ms, "wait_s": QUIET_WAIT_S}),
                       mode_env("score"), QUIET_WAIT_S + GPU_WINDOW_S + 10.0)
    healthy = (isinstance(probe, subprocess.CompletedProcess) and probe.returncode == 0
               and probe.stdout.strip() == "GPU_CHECK_OK")
    failure.detail["recovery"] = {
        "passed": healthy,
        "check_s": round(time.monotonic() - probe_started, 1),
        "reason": ("known-good GPU operation was correct and the GPU was quiet" if healthy else
                   ": ".join(filter(None, (probe.detail["reason"],
                                           probe.detail.get("stderr_tail", "").strip()[-300:])))
                   if isinstance(probe, Verdict) else
                   "GPU check did not return the expected result"),
    }
    if not healthy:
        failure.detail["reason"] += "; GPU recovery check failed; stopping the job"
        return failure
    # A killed worker cannot hand off its cooling debt. Conservatively count
    # its entire elapsed time, including the probe, as GPU work on this rare path.
    from autotuner.measure.session import DUTY_IDLE_FACTOR
    delay = (time.monotonic() - started) * DUTY_IDLE_FACTOR
    ready_at = time.monotonic() + delay
    if not defer_cooling:
        time.sleep(delay)
    return Verdict(False, "timeout", (), {**failure.detail, "abort_job": False},
                   {"cooling_ready_at": ready_at})


def _supervise(command, payload, env, timeout_s, gpu_timeout_s=GPU_WINDOW_S):
    """Monitor a dedicated pipe so cooling and ordinary stdout cannot reset a GPU deadline."""
    read_fd, write_fd = os.pipe()
    proc = None
    try:
        with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout, \
                tempfile.TemporaryFile() as stderr:
            stdin.write(payload.encode())
            stdin.seek(0)
            try:
                proc = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr,
                                        env={**env, "AUTOTUNER_WATCHDOG_FD": str(write_fd)},
                                        pass_fds=(write_fd,))
            except OSError as e:
                return _subprocess_verdict(f"child spawn failed: {e}", "")
            deadline = time.monotonic() + timeout_s
            gpu_deadline = None
            reason = None
            while proc.poll() is None:
                now = time.monotonic()
                if gpu_deadline is not None and now >= gpu_deadline:
                    reason = f"GPU evaluation timeout after {gpu_timeout_s}s"
                    break
                if now >= deadline:
                    reason = f"wall timeout after {timeout_s}s"
                    break
                ready, _, _ = select.select([read_fd], [], [], min(
                    0.05, deadline - now,
                    max(0.0, gpu_deadline - now) if gpu_deadline is not None else 0.05))
                if ready:
                    for event in os.read(read_fd, 4096):
                        if event == ord("B") and gpu_deadline is None:
                            gpu_deadline = time.monotonic() + gpu_timeout_s
                        elif event == ord("E"):
                            gpu_deadline = None
            if reason is not None:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    reason += "; worker did not exit after kill"
            stdout.seek(0)
            stderr.seek(0)
            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            if reason is not None or proc.returncode != 0:
                v = _subprocess_verdict(reason or f"child exit {proc.returncode}", _tail(err))
                v.detail["failure_kind"] = "timeout" if reason is not None else "exit"
                v.detail["gpu_timeout_s"] = gpu_timeout_s
                v.detail["worker_exited"] = proc.poll() is not None
                return v
            return subprocess.CompletedProcess(command, proc.returncode, out, err)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass  # A stuck driver can delay exit even after SIGKILL.
        os.close(read_fd)
        os.close(write_fd)


class WorkerFailed(RuntimeError):
    """The worker exited, could not start, or returned no valid verdict."""


class WorkerUnresponsive(WorkerFailed):
    """Stop orchestration without issuing another GPU command."""


def require_responsive(verdict: Verdict) -> None:
    if verdict.detail.get("abort_job"):
        context = ": ".join(str(verdict.detail[k]) for k in ("kernel_id", "phase")
                            if verdict.detail.get(k))
        message = f"{context}: {verdict.detail['reason']}" if context else verdict.detail["reason"]
        if verdict.detail.get("stderr_tail"):
            message += "\nWorker stderr:\n" + verdict.detail["stderr_tail"]
        error_type = WorkerUnresponsive if verdict.detail.get("failure_kind") == "timeout" else WorkerFailed
        raise error_type(message)


def _subprocess_verdict(reason: str, stderr_tail: str) -> Verdict:
    return Verdict(
        passed=False,
        failed_gate="subprocess",
        gates_passed=(),
        detail={"reason": reason, "stderr_tail": stderr_tail,
                "abort_job": True, "failure_kind": "exit"},
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
