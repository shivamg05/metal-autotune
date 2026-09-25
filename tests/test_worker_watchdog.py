"""Exercise process supervision without wedging the computer's shared GPU."""

import json
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


@pytest.mark.parametrize('recovery', ['success', 'timeout', 'bad_result', 'crash'])
def test_timeout_requires_successful_recovery_before_continuing(monkeypatch, recovery):
    from types import SimpleNamespace
    from autotuner.sandbox import protocol
    calls = []
    failure = Verdict(False, 'subprocess', (), {
        'abort_job': True, 'failure_kind': 'timeout', 'reason': 'GPU evaluation timeout',
        'worker_exited': True})
    responses = [failure, {
        'success': subprocess.CompletedProcess([], 0, 'GPU_CHECK_OK\n', ''),
        'timeout': Verdict(False, 'subprocess', (), {'reason': 'probe timed out'}),
        'bad_result': subprocess.CompletedProcess([], 0, 'wrong', ''),
        'crash': subprocess.CompletedProcess([], 1, '', 'driver error'),
    }[recovery]]
    def supervise(*args):
        calls.append(args)
        return responses.pop(0)
    monkeypatch.setattr(protocol, '_supervise', supervise)
    monkeypatch.setattr(protocol, '_quiet_ms', 6.0)
    spec = SimpleNamespace(to_json=lambda: '{}', kernel={'kernel_id': 'bad'},
                           phase='validate', defer_cooling=True)
    result = protocol.run_job(spec, 'validate', 300)
    assert len(calls) == 2
    assert calls[1][0][-1] == 'autotuner.sandbox.recovery'
    assert json.loads(calls[1][1]) == {'quiet_ms': 6.0, 'wait_s': protocol.QUIET_WAIT_S}
    assert result.detail['kernel_id'] == 'bad'
    assert result.detail['recovery']['passed'] == (recovery == 'success')
    if recovery == 'success':
        require_responsive(result)
        assert not result.passed and result.failed_gate == 'timeout'
        assert result.timing['cooling_ready_at'] >= time.monotonic() - .1
    else:
        with pytest.raises(WorkerUnresponsive):
            require_responsive(result)


def test_worker_must_exit_before_recovery(monkeypatch):
    from types import SimpleNamespace
    from autotuner.sandbox import protocol
    calls = []
    def supervise(*args):
        calls.append(args)
        return Verdict(False, 'subprocess', (), {'abort_job': True,
            'failure_kind': 'timeout', 'reason': 'worker still alive', 'worker_exited': False})
    monkeypatch.setattr(protocol, '_supervise', supervise)
    monkeypatch.setattr(protocol, '_quiet_ms', 6.0)
    spec = SimpleNamespace(to_json=lambda: '{}', kernel={}, phase='validate', defer_cooling=True)
    with pytest.raises(WorkerUnresponsive):
        require_responsive(protocol.run_job(spec, 'validate', 300))
    assert len(calls) == 1


def test_recovered_timeout_rejects_only_that_candidate(monkeypatch):
    from types import SimpleNamespace
    from autotuner.ladder import gates
    monkeypatch.setattr(gates, '_validate', lambda job: None)
    monkeypatch.setattr(gates, 'check', lambda *args: [])
    monkeypatch.setattr(gates, '_spec', lambda job, phase: phase)
    verdicts = iter([
        Verdict(False, 'timeout', (), {'abort_job': False, 'recovery': {'passed': True}},
                {'cooling_ready_at': time.monotonic()}),
        Verdict(True, None, ('compile', 'smoke'), {}, {}),
    ])
    monkeypatch.setattr(gates, 'run_job', lambda *args: next(verdicts))
    job = SimpleNamespace(kernel=None, contract=None, timeout_s=300, run_clock=False)
    assert gates.run_ladder(job).failed_gate == 'timeout'
    assert gates.run_ladder(job).outcome == 'correct_slower'


def test_quiet_reference_is_recorded_once_before_any_candidate(monkeypatch):
    from types import SimpleNamespace
    from autotuner.sandbox import protocol
    calls = []
    def supervise(command, payload, *args):
        calls.append((command[-1], payload))
        if command[-1] == 'autotuner.sandbox.recovery':
            return subprocess.CompletedProcess([], 0, 'GPU_QUIET_MS 6.5\n', '')
        return subprocess.CompletedProcess([], 0, json.dumps(
            {'passed': True, 'failed_gate': None, 'gates_passed': [], 'detail': {}, 'timing': {}}), '')
    monkeypatch.setattr(protocol, '_supervise', supervise)
    monkeypatch.setattr(protocol, '_quiet_ms', None)
    spec = SimpleNamespace(to_json=lambda: '{}', kernel={}, phase='score', defer_cooling=True)
    protocol.run_job(spec, 'score', 300)
    protocol.run_job(spec, 'score', 300)
    assert [c[0] for c in calls] == ['autotuner.sandbox.recovery', 'autotuner.sandbox.worker',
                                     'autotuner.sandbox.worker']
    assert calls[0][1] == '' and protocol._quiet_ms == 6.5


def test_failed_quiet_reference_stops_before_any_candidate(monkeypatch):
    from types import SimpleNamespace
    from autotuner.sandbox import protocol
    monkeypatch.setattr(protocol, '_supervise',
                        lambda *args: subprocess.CompletedProcess([], 1, '', 'Metal device lost'))
    monkeypatch.setattr(protocol, '_quiet_ms', None)
    spec = SimpleNamespace(to_json=lambda: '{}', kernel={}, phase='score', defer_cooling=True)
    with pytest.raises(WorkerFailed, match='Metal device lost'):
        protocol.run_job(spec, 'score', 300)


# A kernel that needs about 8 s of the whole GPU; the worker is killed long before.
SPIN = (
    "import mlx.core as mx\n"
    "k = mx.fast.metal_kernel(name='spin', input_names=['x'], output_names=['y'], source="
    "'uint i = thread_position_in_grid.x; float a = x[i];"
    " for (uint n = 0; n < 40000000u; ++n) { a = a * 1.0000001f + 1e-7f; } y[i] = a;')\n"
    "x = mx.ones((65536,))\n"
    "y = k(inputs=[x], grid=(65536, 1, 1), threadgroup=(256, 1, 1),"
    " output_shapes=[x.shape], output_dtypes=[mx.float32])[0]\n"
    "os.write(fd, b'B'); mx.eval(y)\n"
)

TIMED_OUT = {'reason': 'GPU evaluation timeout', 'abort_job': True, 'failure_kind': 'timeout'}


def test_recovery_waits_for_a_killed_workers_kernel_to_leave_the_gpu(monkeypatch):
    """On the real GPU, a killed worker's kernel keeps running. Recovery must
    wait until it is gone, and must fail if the GPU never gets quiet."""
    from autotuner.sandbox import protocol
    monkeypatch.setattr(protocol, '_quiet_ms', None)
    quiet_ms = protocol._quiet_reference()
    clean = protocol._recover_timeout(Verdict(False, 'subprocess', (), TIMED_OUT),
                                      time.monotonic(), quiet_ms, defer_cooling=True)
    assert clean.detail['recovery']['passed'], clean.detail['recovery']
    killed = worker(SPIN, timeout=30, gpu_timeout=0.5)
    assert killed.detail['failure_kind'] == 'timeout' and killed.detail['worker_exited']
    result = protocol._recover_timeout(killed, time.monotonic(), quiet_ms, defer_cooling=True)
    assert result.detail['recovery']['passed'], result.detail['recovery']
    assert not result.detail['abort_job']
    # the same check on an idle GPU is startup plus two one-second readings
    assert result.detail['recovery']['check_s'] >= clean.detail['recovery']['check_s'] + 3
    monkeypatch.setattr(protocol, 'QUIET_WAIT_S', 2.0)
    busy = protocol._recover_timeout(Verdict(False, 'subprocess', (), TIMED_OUT),
                                     time.monotonic(), quiet_ms / 10, defer_cooling=True)
    assert not busy.detail['recovery']['passed']
    assert 'GPU still busy' in busy.detail['recovery']['reason']
    with pytest.raises(WorkerUnresponsive):
        require_responsive(busy)
