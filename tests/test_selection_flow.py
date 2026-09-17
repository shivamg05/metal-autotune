"""Queue and calibration failure paths, with no GPU execution."""

from types import SimpleNamespace

import pytest

from autotuner import loop
from autotuner.measure.clocks import compare, comparison_from_samples
from autotuner.regions.types import Region, Stretch


def region(name, *positions):
    return Region(name, ("mx.exp",), [
        Stretch("w", pos, pos, (2 * pos,), (2 * pos + 1,), ("@0",))
        for pos in positions])


def runner_stub(tmp_path, monkeypatch):
    runner = object.__new__(loop.JobRunner)
    runner.manifest = SimpleNamespace(budget_total=10)
    runner.total_hypotheses = 0
    runner.selection_wave = 0
    runner.pending_regions = []
    runner.traces, runner.step_ms, runner.peaks = {}, {}, None
    runner.work_dir = tmp_path
    runner.events = []
    runner.log = SimpleNamespace(append=lambda kind, **row: runner.events.append((kind, row)))
    runner.report = SimpleNamespace(coverage={"selection": {"waves": []}}, session={},
                                    write=lambda path: None)
    runner.tracer = SimpleNamespace(
        install=lambda: runner.events.append(("install", {})),
        uninstall=lambda: runner.events.append(("uninstall", {})))
    runner._capture = lambda regions: runner.events.append(("capture", regions.copy()))
    monkeypatch.setattr(loop, "estimate_regions", lambda *args: {})
    return runner


def test_rejected_alternatives_do_not_exhaust_pricing_while_search_work_waits(tmp_path, monkeypatch):
    runner = runner_stub(tmp_path, monkeypatch)
    runner.pending_regions = [region(name, 0) for name in ("a", "b", "c")]
    runner._price_and_rank = lambda regions: []

    assert runner._next_regions(blockers=[region("already_priced", 10)]) == []
    assert runner.selection_wave == 1
    assert [r.fingerprint for r in runner.pending_regions] == ["b", "c"]


def test_empty_search_queue_still_reaches_deferred_viable_regions(tmp_path, monkeypatch):
    runner = runner_stub(tmp_path, monkeypatch)
    runner.pending_regions = [region(name, 0) for name in ("a", "b", "c")]
    runner._price_and_rank = lambda regions: regions if regions[0].fingerprint == "c" else []

    assert [r.fingerprint for r in runner._next_regions()] == ["c"]
    assert runner.selection_wave == 3
    assert not runner.pending_regions


def test_trimmed_representative_is_recaptured_before_cached_region_is_repriced(tmp_path, monkeypatch):
    runner = runner_stub(tmp_path, monkeypatch)
    candidate, unchanged = region("trimmed", 0, 4), region("unchanged", 8)
    pending = region("pending", 0, 6)
    runner.pending_regions = [pending]
    saved_ids = {"trimmed": (0,), "unchanged": (16,)}

    def capture(regions):
        assert regions == [candidate]
        for r in regions:
            saved_ids[r.fingerprint] = r.members[0].input_ids
        runner.events.append(("capture", regions.copy()))

    def price(regions):
        for r in regions:
            # BoundaryStore binds by trace array ID, not input position.
            assert saved_ids[r.fingerprint] == r.members[0].input_ids
        runner.events.append(("price", regions.copy()))
        return regions

    runner._capture, runner._price_and_rank = capture, price
    assert runner._refresh_after_ship([candidate, unchanged], [region("shipped", 0)]) == \
        [candidate, unchanged]
    assert candidate.members[0].start_seq == 4
    assert pending.members[0].start_seq == 6
    assert [kind for kind, _ in runner.events] == ["install", "capture", "uninstall", "price"]


def test_recapture_failure_restores_tracer(tmp_path, monkeypatch):
    runner = runner_stub(tmp_path, monkeypatch)
    runner._capture = lambda regions: (_ for _ in ()).throw(RuntimeError("capture failed"))
    with pytest.raises(RuntimeError, match="capture failed"):
        runner._refresh_after_ship([region("trimmed", 0, 4)], [region("shipped", 0)])
    assert [kind for kind, _ in runner.events] == ["install", "uninstall"]


def test_fully_covered_regions_are_removed_without_recapture(tmp_path, monkeypatch):
    runner = runner_stub(tmp_path, monkeypatch)
    runner.pending_regions = [region("pending", 0)]
    assert runner._refresh_after_ship([region("ranked", 0)], [region("shipped", 0)]) == []
    assert runner.pending_regions == []
    assert [kind for kind, _ in runner.events] == ["region_covered", "region_covered"]


class PositionBiasedSession:
    def __init__(self, factors):
        self.factors, self.calls = factors, 0

    def fresh_chunk(self, fn):
        pass

    def warm_until_stable(self, fn):
        pass

    def timed(self, fn):
        value = 0.1 * self.factors[self.calls % len(self.factors)]
        self.calls += 1
        return value

    def settle(self):
        pass

    def log(self, *args, **kwargs):
        pass


def test_clock_balances_repeated_outer_slot_bias():
    session = PositionBiasedSession((1.02, 1.0, 1.0, 1.02))
    result = compare(session, lambda: None, lambda: None, pairs=16)
    assert result.median_delta_ms == pytest.approx(0.0)
    assert not result.wins_by(0.0)
    assert not result.loses_by(0.0)


@pytest.mark.parametrize("recovery", [False, True])
def test_failed_aa_control_gets_one_retry_then_stops_or_recovers(tmp_path, monkeypatch, recovery):
    runner = runner_stub(tmp_path, monkeypatch)
    # Alternating block bias can fool even balanced ordering; the null check
    # must reject that session instead of calling it harmless noise.
    suspect = compare(PositionBiasedSession((1.02, 1.0, 1.0, 1.02, 1.0, 1.02, 1.02, 1.0)),
                      lambda: None, lambda: None, pairs=8)
    assert suspect.wins_by(0.0)
    healthy = comparison_from_samples([100.0] * 4, [100.0] * 4)
    observations = []

    def aa(session, pairs):
        observations.append(pairs)
        return healthy if recovery and len(observations) == 2 else suspect

    runner.gpu_busy_at_start, runner.session = None, object()
    runner._record_peaks = lambda *args: None
    runner._env_warning = lambda detail: runner.events.append(("warning", detail))
    monkeypatch.setattr(loop, "measure_peaks", lambda session: None)
    monkeypatch.setattr(loop, "peaks_implausible", lambda peaks: None)
    monkeypatch.setattr(loop, "aa_null", aa)

    if recovery:
        runner.measure_machine()
    else:
        with pytest.raises(RuntimeError, match="A/A timing control failed twice"):
            runner.measure_machine()
    assert observations == [8, 8]
    assert runner.report.session["aa_control"] == {"passed": recovery, "attempts": 2}
