"""Exercise process supervision without wedging the computer's shared GPU."""

import os
import subprocess
import sys
import time

import pytest

from autotuner.sandbox.protocol import Verdict, WorkerFailed, WorkerUnresponsive, _supervise, require_responsive

PREAMBLE = "import os, time; fd = int(os.environ['AUTOTUNER_WATCHDOG_FD']); "


def worker(code, *, timeout=3, gpu_timeout=0.15):
    return _supervise([sys.executable, "-c", PREAMBLE + code], "{}", os.environ,
                      timeout, gpu_timeout)


def test_stalled_evaluation_stops_without_waiting_for_job_budget():
    started = time.monotonic()
    result = worker("os.write(fd, b'B'); time.sleep(30)")
    assert time.monotonic() - started < 2
    assert isinstance(result, Verdict) and result.detail["abort_job"]
    assert "GPU evaluation timeout" in result.detail["reason"]
    with pytest.raises(WorkerUnresponsive, match="GPU evaluation timeout"):
        require_responsive(result)


def test_nested_starts_cannot_extend_a_stalled_evaluation():
    result = worker("os.write(fd, b'B'); time.sleep(0.1); os.write(fd, b'B'); time.sleep(0.1)")
    assert isinstance(result, Verdict) and result.detail["abort_job"]


def test_cooling_between_evaluations_is_outside_gpu_deadline():
    result = worker("os.write(fd, b'BE'); time.sleep(0.3); os.write(fd, b'BE'); print('done')")
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 0 and result.stdout.strip() == "done"


def test_large_output_cannot_block_deadline_monitor():
    result = worker("os.write(fd, b'B'); print('x' * 1000000, flush=True); time.sleep(30)")
    assert isinstance(result, Verdict) and result.detail["abort_job"]


def test_overall_budget_still_limits_a_worker_outside_gpu_work():
    result = worker("time.sleep(30)", timeout=0.1)
    assert isinstance(result, Verdict) and "wall timeout" in result.detail["reason"]


def test_crash_during_evaluation_stops_the_job():
    result = worker("os.write(fd, b'B'); time.sleep(0.1); os._exit(9)")
    assert isinstance(result, Verdict) and result.detail["abort_job"]


def test_normal_evaluation_failure_does_not_stop_the_job():
    require_responsive(Verdict(False, "smoke", (), {"reason": "wrong output"}))


def test_worker_traceback_survives_abort_for_job_report():
    result = worker("raise ValueError('broken candidate metadata')")
    assert isinstance(result, Verdict)
    with pytest.raises(WorkerFailed) as error:
        require_responsive(result)
    assert not isinstance(error.value, WorkerUnresponsive)
    assert "child exit 1" in str(error.value)
    assert "ValueError: broken candidate metadata" in str(error.value)


def test_ladder_stops_before_scoring_after_worker_timeout(monkeypatch):
    from types import SimpleNamespace
    from autotuner.ladder import gates

    calls = []
    monkeypatch.setattr(gates, "_validate", lambda job: None)
    monkeypatch.setattr(gates, "check", lambda *args: [])
    monkeypatch.setattr(gates, "_spec", lambda job, phase: phase)

    def stalled(spec, mode, timeout):
        calls.append(mode)
        return Verdict(False, "subprocess", (), {"abort_job": True, "failure_kind": "timeout", "reason": "GPU timeout"})

    monkeypatch.setattr(gates, "run_job", stalled)
    with pytest.raises(WorkerUnresponsive):
        gates.run_ladder(SimpleNamespace(kernel=None, contract=None, timeout_s=300))
    assert calls == ["validate"]
