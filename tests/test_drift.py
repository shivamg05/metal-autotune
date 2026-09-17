"""Measurement on a machine whose GPU speed moves during the run.

Every number this project reports is a ratio of two timings: a region's share
of the step, a kernel's win over the library, a region's distance from its
roofline, the step before against the step after. Whether that ratio is true
depends entirely on whether its two halves were measured in the same window.

These tests run the same drifting machine past both ways of getting each
number, so the difference between them is visible rather than argued.
"""

import statistics

import pytest

from autotuner.measure.clocks import compare
from tests.conftest import tracer_for_module
from tests.drift import DriftingSession, costed, flip, ramp

_module_tracer = tracer_for_module()

BASELINE_S = 0.002   # 2 ms on a healthy machine
CANDIDATE_S = 0.001  # 1 ms: a true 2x win
TRUE_RATIO = CANDIDATE_S / BASELINE_S


def test_measuring_one_arm_then_the_other_hides_a_2x_win():
    """The antipattern, and why this file exists. Time 32 baselines, then 32
    candidates, on a machine that slows as it goes: the second arm pays a
    penalty the first one never saw, and a real 2x win reports as noise."""
    session = DriftingSession(ramp(1.0, 4.0, over=60))
    base_fn, cand_fn = costed(BASELINE_S), costed(CANDIDATE_S)
    base = [session.timed(base_fn) for _ in range(32)]
    cand = [session.timed(cand_fn) for _ in range(32)]

    naive_ratio = statistics.median(cand) / statistics.median(base)
    assert naive_ratio > 0.75, (
        f"expected the sequential method to lose the win, got {naive_ratio:.3f}"
    )
    # the truth is 0.5; anything above 0.75 has thrown away half the speedup
    assert abs(naive_ratio - TRUE_RATIO) > 0.25


def test_interleaved_comparison_recovers_the_win_on_the_same_machine():
    """Same drift, same functions, alternated ABBA: the ratio survives."""
    session = DriftingSession(ramp(1.0, 4.0, over=60))
    result = compare(session, costed(BASELINE_S), costed(CANDIDATE_S), pairs=32)

    assert result.median_ratio == pytest.approx(TRUE_RATIO, rel=0.02)
    # and the absolute times it was computed from really did move
    assert max(result.baseline_ms) / min(result.baseline_ms) > 2.0


def test_interleaved_comparison_survives_power_state_flips():
    """The reference machine's actual behaviour: two discrete speeds, flipping
    during the measurement. Harder than a ramp, because a flip can land inside
    a pair rather than between pairs."""
    session = DriftingSession(flip(low=1.0, high=2.5, every=17))
    result = compare(session, costed(BASELINE_S), costed(CANDIDATE_S), pairs=32)

    assert result.median_ratio == pytest.approx(TRUE_RATIO, rel=0.05)
    assert max(result.baseline_ms) / min(result.baseline_ms) > 2.0


def test_an_unchanged_model_reports_no_speedup_when_the_headline_is_paired():
    """The 2254 run reported the decode step going 52.107 -> 48.294 ms with
    nothing installed. Same model both times; the only difference was that the
    'before' clock was taken 19 s into the job on a machine still settling from
    boot and the 'after' clock 36 minutes later on a quiet one."""
    settling = ramp(1.10, 1.0, over=40)  # the machine gets 10% faster as it settles
    unchanged = costed(0.050)

    session = DriftingSession(settling)
    before = statistics.median([session.timed(unchanged) for _ in range(9)])
    for _ in range(200):
        session.timed(unchanged)  # the job runs
    after = statistics.median([session.timed(unchanged) for _ in range(9)])
    assert before / after > 1.05, "expected a phantom speedup from settling alone"

    paired = compare(DriftingSession(settling), costed(0.050), costed(0.050), pairs=32)
    assert paired.median_ratio == pytest.approx(1.0, rel=0.01)


def test_pricing_a_region_on_a_slow_chip_does_not_inflate_its_headroom():
    """Exercise the actual shared clock with known costs and a controlled
    speed change. Multiplying two separate live GPU runs by synthetic factors
    also imports their unrelated hardware noise, so it cannot isolate this
    property. Live replay and capture are covered in test_regions.
    """
    from autotuner.measure.clocks import sample_group
    from autotuner.regions.price import _group_price
    from tests.drift import steady

    arms = {"step": costed(0.100), "r:library": costed(0.024),
            "r:link": costed(0.004), "r:probe": costed(0.010)}

    def price_at(slowdown):
        rows = sample_group(DriftingSession(steady(slowdown)), arms, pairs=8)
        return _group_price(rows, "r", iters=10, linked=True)

    healthy, slow = price_at(1.0), price_at(3.0)

    assert healthy.ms == pytest.approx(2.0)
    assert slow.ms / healthy.ms == pytest.approx(3.0)
    assert healthy.share == pytest.approx(0.02)
    assert slow.share == pytest.approx(healthy.share)
    assert slow.ms / slow.floor_ms == pytest.approx(healthy.ms / healthy.floor_ms)


def test_abba_centers_a_linear_drift_before_deciding_an_unchanged_model_won():
    result = compare(DriftingSession(ramp(1.0, 4.0, over=32)),
                     costed(0.028), costed(0.028), pairs=16)

    assert result.n == 8  # eight independent ABBA blocks
    assert result.median_ratio == pytest.approx(1.0, abs=1e-12)
    assert result.median_delta_ms == pytest.approx(0.0, abs=1e-12)
    assert result.sigma_ms < 1e-12
    assert not result.wins_by(1e-9)
    assert not result.loses_by(1e-9)


@pytest.mark.parametrize("ratio", [0.97, 1.03])
def test_abba_resolves_a_three_percent_change_during_linear_drift(ratio):
    result = compare(DriftingSession(ramp(1.0, 4.0, over=32)),
                     costed(0.028), costed(0.028 * ratio), pairs=16)

    assert result.n == 8
    assert result.median_ratio == pytest.approx(ratio, abs=1e-12)
    assert result.wins_by(0.0) == (ratio < 1.0)
    assert result.loses_by(0.0) == (ratio > 1.0)


def test_stability_is_high_when_pairs_agree_and_low_when_they_do_not():
    """The score every recorded number carries: how much the per-pair ratios
    agreed. It reads the ratios, not the absolute times, because on a drifting
    machine the absolutes are supposed to move and the ratios are not."""
    steady = compare(DriftingSession(ramp(1.0, 4.0, over=60)),
                     costed(BASELINE_S), costed(CANDIDATE_S), pairs=32)
    # The inner slots run slowly. Alternating ABBA/BAAB gives that disadvantage
    # to different arms, so the block ratios disagree and must show low stability.
    alternating_advantage = lambda call: (1.0, 3.0, 3.0, 1.0)[call % 4]
    churn = compare(DriftingSession(alternating_advantage),
                    costed(BASELINE_S), costed(CANDIDATE_S), pairs=32)

    assert steady.stability > 0.9
    assert churn.stability < 0.5
    assert 0.0 <= churn.stability <= 1.0
