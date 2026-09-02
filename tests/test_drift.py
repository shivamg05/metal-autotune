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
    """s_max divides a region's cost by a roofline built from peaks measured at
    job start. The cost used to be an absolute region clock taken whenever that
    region's turn came round, so a region priced while the chip was throttled
    looked that many times further from its roofline than it is.

    Price the same region twice, on a healthy chip and on one running at a
    third speed, and check both numbers the loop could feed the roofline: the
    absolute clock triples, the measured share does not.
    """
    from autotuner.regions.build import build_stretches
    from autotuner.regions.price import capture_boundaries, price_region
    from autotuner.regions.fingerprint import group_copies
    from tests.conftest import require_healthy_gpu, require_quiet_load
    from tests.test_regions import traced, tracer
    from tests.drift import steady

    # real replay on the real GPU under a simulated slowdown; a machine that is
    # already throttled or loaded cannot supply the healthy half of the pair
    require_healthy_gpu()
    require_quiet_load()
    model, x, trace = traced("repeated_layers", (4096, 16))
    region_of = lambda: max(
        (r for r in group_copies({"w": trace}, {"w": build_stretches(trace, "w")})
         if r.copies == 4),
        key=lambda r: len(r.ops),
    )
    rep = region_of().members[0]
    arrays = capture_boundaries(tracer(), model, [x], trace,
                                set(rep.input_ids) | set(rep.output_ids))
    weights = {a: arrays[a] for a in rep.input_ids if a in trace.weights}
    inputs = {a: arrays[a] for a in rep.input_ids if a not in trace.weights}

    def price_at(chip_speed):
        region = region_of()
        price_region(region, DriftingSession(steady(chip_speed)), {"w": trace},
                     {("w", rep.start_seq): [inputs]}, {"w": weights},
                     {"w": lambda: model(x)})
        return region

    healthy, slow = price_at(1.0), price_at(3.0)

    # the absolute clock tracks the chip, which is why feeding it to the
    # roofline reported headroom that was really just a slow moment
    assert slow.t_rep_ms["w"] / healthy.t_rep_ms["w"] == pytest.approx(3.0, rel=0.3)
    # the share does not, which is why the roofline is fed from it instead
    assert slow.p_rep["w"] == pytest.approx(healthy.p_rep["w"], rel=0.25)
    # and a share is a share: four copies of a region inside the model cannot
    # together cost more than the step that contains them. The 8B decode run
    # reported 1.756 for one region, and 586 ms of regions against a 52 ms step.
    for region in (healthy, slow):
        assert 0 < region.p["w"] <= 1.0, region.p


def test_stability_is_high_when_pairs_agree_and_low_when_they_do_not():
    """The score every recorded number carries: how much the per-pair ratios
    agreed. It reads the ratios, not the absolute times, because on a drifting
    machine the absolutes are supposed to move and the ratios are not."""
    steady = compare(DriftingSession(ramp(1.0, 4.0, over=60)),
                     costed(BASELINE_S), costed(CANDIDATE_S), pairs=32)
    # every second call changes speed, so pairs straddle a flip and disagree
    churn = compare(DriftingSession(flip(low=1.0, high=3.0, every=1)),
                    costed(BASELINE_S), costed(CANDIDATE_S), pairs=32)

    assert steady.stability > 0.9
    assert churn.stability < 0.5
    assert 0.0 <= churn.stability <= 1.0
