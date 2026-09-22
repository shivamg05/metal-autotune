"""Clock protocol and application-loop contract, independent of GPU noise."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from autotuner.e2e import E2EResult
from autotuner.loop import JobRunner
from autotuner.measure.clocks import comparison_from_samples
from autotuner.measure.sequences import compare_sequences
from autotuner_runtime.sequence import make_sequence


def test_resolved_small_gain_can_ship_but_noise_and_regression_cannot():
    runner = object.__new__(JobRunner)
    def result(samples):
        clock = comparison_from_samples([1000.0] * 10, samples)
        return E2EResult(checks=[SimpleNamespace(passed=True)], veto=clock,
                         workload_vetos={"main": clock})
    # A 0.29% improvement was rejected in the FLUX run only by the old 0.5% gate.
    assert runner._model_win(result([997.1 + i % 2 * .1 for i in range(10)]))
    assert not runner._model_win(result([995.0, 1004.0] * 5))
    assert not runner._model_win(result([1001.0] * 10))
    bad_outputs = result([990.0] * 10)
    bad_outputs.checks[0].passed = False
    assert not runner._model_win(bad_outputs)


def test_independent_confirmation_does_not_reuse_nomination_or_retry(monkeypatch):
    from autotuner import loop
    runner = object.__new__(JobRunner)
    runner.session = object()
    runner.log = SimpleNamespace(append=lambda *a, **k: None)
    runner._timed_arms = lambda incumbent: {"main": ("fresh_reference", "fresh_candidate")}
    nomination_clock = comparison_from_samples([100.0] * 10, [99.0] * 10)
    nomination = E2EResult(veto=nomination_clock, workload_vetos={"main": nomination_clock})
    calls = []
    def compare(session, baseline, candidate, pairs, **kwargs):
        calls.append((baseline, candidate, pairs))
        return comparison_from_samples([100.0] * 10, [100.0] * 10)
    monkeypatch.setattr(loop, "compare", compare)
    confirmed = runner._confirm_model_win(nomination, object())
    assert runner._model_win(nomination)
    assert not runner._model_win(confirmed)
    assert calls == [("fresh_reference", "fresh_candidate", loop.SHIP_PAIRS)]


def test_sequences_balance_order_and_cool_only_outside_complete_runs():
    events = []
    class Session:
        def settle(self): events.append("cool")
        def warm_until_stable(self, fn):
            events.append("calibrate")
            return [0.1] * (4 if fn.__name__ == "warm_a" else 6)
        def timed(self, fn): return fn()
        def log(self, *args, **kwargs): pass
    def warm_a(): events.append("warm_a"); return .1
    def warm_b(): events.append("warm_b"); return .1
    def a(): events.extend(["a_step"] * 20); return 20.0
    def b(): events.extend(["b_step"] * 20); return 18.0
    result = compare_sequences(Session(), a, b, warm_a, warm_b, pairs=4, warmup_steps=3)
    assert result["warmup_steps"] == 7
    assert result["baseline_sequence_ms"] == 20000
    assert result["candidate_sequence_ms"] == 18000
    assert result["win_confirmed"]
    assert [o["arm"] for o in result["observations"]] == ["baseline", "candidate", "candidate", "baseline"] * 2
    timed_events = events[5:]  # initial settle and two calibrate/settle pairs
    expected = []
    for arm in ["a", "b", "b", "a"] * 2:
        expected.extend([f"warm_{arm}"] * 7 + [f"{arm}_step"] * 20 + ["cool"])
    assert timed_events == expected + ["cool"]  # and whatever debt is left at the end


def test_short_sequences_run_back_to_back_and_cool_once_at_the_end():
    """A 20 ms run heats nothing, and an idle before the next one puts it on
    ramped-down clocks: a kernel 1.4x faster back to back read slower."""
    events = []
    class Session:
        def settle(self): events.append("cool")
        def warm_until_stable(self, fn): return [0.02]
        def timed(self, fn): return fn()
        def log(self, *args, **kwargs): pass
    def run(): events.append("run"); return 0.02
    compare_sequences(Session(), run, run, lambda: 0.02, lambda: 0.02, pairs=4, warmup_steps=1)
    assert events == ["cool"] + ["run"] * 8 + ["cool"]  # prior debt first, then nothing until the end


def test_sequence_exception_still_pays_prior_work_debt():
    events = []
    class Session:
        def settle(self): events.append("cool")
        def warm_until_stable(self, fn): return [.1]
        def timed(self, fn): return fn()
        def log(self, *args, **kwargs): pass
    def fail(): raise RuntimeError("device failed")
    with pytest.raises(RuntimeError, match="device failed"):
        compare_sequences(Session(), fail, fail, lambda: .1, lambda: .1)
    assert events[-1] == "cool"


def test_a_sequence_chains_every_step_and_returns_all_outputs():
    """One graph of dependent steps: each call after the first takes an input
    derived from the previous output, and every output comes back, so the
    caller's eval reaches all of them and none can be skipped as unused."""
    calls = []
    def step(x, w):
        calls.append(x)
        return x * w
    x, w = mx.array([7.0]), mx.array([2.0, 2.0])
    outs = make_sequence(step, [x, w], 5)()
    mx.eval(outs)
    assert len(calls) == 5 and len(outs) == 5
    assert calls[0] is x and all(c is not x for c in calls[1:])  # linked to the last output
    assert all(o.tolist() == [14.0, 14.0] for o in outs)


def _fake_session(durations, slept):
    """A BenchSession whose clock reads each timed() call as the next duration."""
    from autotuner_runtime.sequence import BenchSession
    ticks = []
    now = 0.0
    for d in durations:
        ticks += [now, now + d]
        now += d
    reads = iter(ticks)
    return BenchSession(clock=lambda: next(reads), sleep=slept.append)


def test_bench_session_warms_until_the_reading_stops_falling():
    slept = []
    durations = [0.020, 0.010, 0.008, 0.00795] + [0.00795] * 9 + [0.001] * 5  # never reached
    session = _fake_session(durations, slept)
    calls = []
    kept = session.warm_until_stable(lambda: calls.append(1))
    # the first call is thrown away; 0.008 improves on 0.010 by more than 1%; the
    # 0.00795 readings improve by 50 us, under max(1%, 100 us), and after ten of
    # them warming stops without reading the fast tail
    assert kept == pytest.approx([0.010, 0.008] + [0.00795] * 10)
    assert len(calls) == 13
    session.settle()
    assert slept == [pytest.approx(3.0 * sum(durations[:13]))]
    session.settle()
    assert len(slept) == 1  # debt was cleared


def test_bench_session_caps_a_reading_that_keeps_falling():
    slept = []
    durations = [1.0 * 0.98 ** i for i in range(80)]
    session = _fake_session(durations, slept)
    kept = session.warm_until_stable(lambda: None)
    assert len(kept) == 60 and kept == pytest.approx(durations[1:61])
