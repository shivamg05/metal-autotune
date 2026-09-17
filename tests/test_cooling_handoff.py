"""Cooling follows the GPU across worker exits, failures and CPU preparation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from autotuner.measure.session import Session
from autotuner.sandbox.protocol import LadderSpec, Verdict


def spec():
    return LadderSpec(kernel={}, assoc_tag="preserving", nodes_json="[]",
        input_ids=(), output_ids=(), eval_sets=(), tolerances={}, kappa=1.25,
        changing_floor=None, min_win_ms=0, phase="validate", defer_cooling=True)


def clock_session():
    now, sleeps = [100.], []
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    return Session(now=lambda: now[0], sleep=sleep), now, sleeps


@pytest.mark.parametrize("elapsed,remaining", [(0, 30), (12, 18), (40, 0)])
def test_worker_deadline_is_not_restarted_in_parent(elapsed, remaining):
    parent, now, sleeps = clock_session()
    now[0] += elapsed  # process exit, parsing or judge thinking
    parent.adopt_cooling(130.)
    parent.wait_ready()
    assert sleeps == ([remaining] if remaining else [])
    assert not parent.wait_ready()


def test_handoff_cannot_shorten_existing_cooling():
    parent, _, sleeps = clock_session()
    parent.adopt_cooling(130.)
    parent.adopt_cooling(110.)
    parent.wait_ready()
    assert sleeps == [30.]


@pytest.mark.parametrize("bad", [None, True, -1, float("nan"), float("inf"), "130"])
def test_invalid_handoff_is_rejected(bad):
    with pytest.raises(ValueError):
        Session().adopt_cooling(bad)


@pytest.mark.parametrize("failed", [False, True])
def test_completed_worker_hands_off_even_a_failed_gate(monkeypatch, failed):
    from autotuner.ladder import child
    session, now, sleeps = clock_session()
    monkeypatch.setattr(child, "Session", lambda: session)
    monkeypatch.setattr(child.time, "perf_counter", lambda: now[0])
    def evaluate(spec, session):
        now[0] += 2
        return Verdict(not failed, "smoke" if failed else None, ())
    monkeypatch.setattr(child, "_evaluate_ladder", evaluate)
    result = child.evaluate_ladder(spec())
    assert result.passed is not failed
    assert result.timing["cooling_ready_at"] == 108.
    assert result.timing["pacing_work_s"] == 2.
    assert sleeps == []
    session.wait_ready()
    assert sleeps == [6.]


def test_unexpected_worker_exception_pays_cooling_locally(monkeypatch):
    from autotuner.ladder import child
    session, now, sleeps = clock_session()
    monkeypatch.setattr(child, "Session", lambda: session)
    monkeypatch.setattr(child.time, "perf_counter", lambda: now[0])
    def broken(spec, session):
        now[0] += 2
        raise ValueError("no verdict")
    monkeypatch.setattr(child, "_evaluate_ladder", broken)
    with pytest.raises(ValueError, match="no verdict"):
        child.evaluate_ladder(spec())
    assert sleeps == [6.]


@pytest.mark.parametrize("failed", [False, True])
def test_ladder_waits_between_workers_and_returns_final_deadline(monkeypatch, failed):
    from autotuner.ladder import gates
    parent, now, sleeps = clock_session()
    monkeypatch.setattr(gates, "_validate", lambda job: None)
    monkeypatch.setattr(gates, "check", lambda *args: [])
    def prepare(job, mode):
        now[0] += 2  # trusted spec construction overlaps preceding cooling
        return replace(spec(), phase=mode)
    monkeypatch.setattr(gates, "_spec", prepare)
    phases = []
    def run(spec, mode, timeout):
        assert spec.defer_cooling
        if phases:
            assert now[0] >= 112.  # validation's deadline
        phases.append(mode)
        return Verdict(not failed, "smoke" if failed else None, (),
            {"ship": False}, {"cooling_ready_at": now[0] + 10})
    monkeypatch.setattr(gates, "run_job", run)
    job = SimpleNamespace(kernel=None, contract=None, timeout_s=30, run_clock=True)
    result = gates.run_ladder(job, session=parent)
    assert result.outcome == ("failed" if failed else "correct_slower")
    assert phases == (["validate"] if failed else ["validate", "score"])
    assert sleeps == ([] if failed else [8.])
    # The caller gets the verdict before the final cooling wait.
    parent.wait_ready()
    assert sleeps[-1] == 10.


def test_off_clock_can_return_result_while_cooling_is_pending(monkeypatch):
    import autotuner.measure.session as module
    session, now, sleeps = clock_session()
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    def check():
        now[0] += 2
        return "correctness verdict"
    assert session.off_clock(check, defer_cooling=True) == "correctness verdict"
    assert sleeps == []
    now[0] += 4
    session.wait_ready()
    assert sleeps == [2.]


def test_standalone_worker_still_finishes_cooling(monkeypatch):
    from autotuner.ladder import child
    session, now, sleeps = clock_session()
    monkeypatch.setattr(child, "Session", lambda: session)
    monkeypatch.setattr(child.time, "perf_counter", lambda: now[0])
    def evaluate(spec, session):
        now[0] += 2
        return Verdict(True, None, ())
    monkeypatch.setattr(child, "_evaluate_ladder", evaluate)
    result = child.evaluate_ladder(replace(spec(), defer_cooling=False))
    assert sleeps == [6.]
    assert "cooling_ready_at" not in result.timing


def test_protocol_defaults_to_blocking_for_existing_callers():
    import json
    encoded = json.loads(spec().to_json())
    del encoded["defer_cooling"]
    assert not LadderSpec.from_json(json.dumps(encoded)).defer_cooling
    assert LadderSpec.from_json(spec().to_json()).defer_cooling


@pytest.mark.parametrize("entry", ["timed", "off_clock"])
def test_model_execution_waits_for_worker_cooling(entry):
    import mlx.core as mx
    session, now, sleeps = clock_session()
    session.adopt_cooling(110.)
    def gpu():
        assert now[0] >= 110.
        return mx.ones((1,))
    getattr(session, entry)(gpu)
    assert sleeps[0] == 10.


def test_interrupted_handoff_wait_retains_remaining_time():
    session, now, sleeps = clock_session()
    normal_sleep = session._sleep
    def interrupted(seconds):
        now[0] += 4
        raise InterruptedError("resume later")
    session.adopt_cooling(110.)
    session._sleep = interrupted
    with pytest.raises(InterruptedError):
        session.wait_ready()
    session._sleep = normal_sleep
    session.wait_ready()
    assert sleeps == [6.]
