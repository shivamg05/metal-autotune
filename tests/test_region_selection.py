"""Selection and Amdahl ranking from trace metadata, without running the GPU."""

import pytest

from autotuner.measure.peaks import Peaks
from autotuner.regions.rank import RegionEstimate, apply_floor, estimate_regions, rank, select_frontier
from autotuner.regions.types import Region, Roofline, Stretch
from autotuner.trace.types import Trace, TraceNode


def roof(speedup, bound="memory"):
    return Roofline(1.0, 0.1, 0.01, 1.0, bound, speedup)


def region(name, start, end, workload="w"):
    return Region(name, ("mx.exp",) * (end - start + 1),
                  [Stretch(workload, start, end, (start,), (end + 1,), ("@0",))])


def test_measured_amdahl_ranks_savings_over_region_size_or_local_speedup():
    big = region("big", 0, 0)
    big.p = {"w": 0.60}
    big.roofline = roof(1.2)
    medium = region("medium", 1, 1)
    medium.p = {"w": 0.30}
    medium.roofline = roof(3.0)
    tiny = region("tiny", 2, 2)
    tiny.p = {"w": 0.02}
    tiny.roofline = roof(100.0)

    assert [r.fingerprint for r in rank([big, tiny, medium])] == ["medium", "big", "tiny"]
    assert medium.combined_removable_p == pytest.approx(0.20)
    # The ordering agrees with Amdahl's optimistic whole-step speedup.
    assert 1 / (1 - medium.combined_removable_p) == pytest.approx(1.25)


def test_workloads_use_their_own_headroom():
    mixed = region("mixed", 0, 0, "prefill")
    mixed.members.append(Stretch("decode", 0, 0, (0,), (1,), ("@0",)))
    mixed.p = {"prefill": 0.40, "decode": 0.10}
    mixed.roofline = roof(10.0)  # representative must not overwrite workload prices
    mixed.rooflines = {"prefill": roof(1.0), "decode": roof(2.0)}
    other = region("other", 1, 1)
    other.p = {"w": 0.15}
    other.roofline = roof(2.0)

    assert mixed.removable_p == pytest.approx({"prefill": 0.0, "decode": 0.05})
    assert mixed.combined_removable_p == pytest.approx(0.05)
    assert rank([mixed, other]) == [other, mixed]
    assert apply_floor([mixed]) == [mixed]  # decode can still be specialized


def test_no_headroom_is_zero_savings_not_a_rejection():
    flat = region("flat", 0, 0)
    flat.p = {"w": 0.50}
    flat.roofline = roof(0.9)
    other = region("other", 1, 1)
    other.p = {"w": 0.05}
    other.roofline = roof(2.0)
    assert flat.combined_removable_p == 0.0
    assert apply_floor([flat, other]) == [flat, other]
    assert rank([flat, other]) == [other, flat]  # a small real win beats a large empty one


def test_existing_rejection_is_preserved():
    flat = region("flat", 0, 0)
    flat.p = {"w": 0.50}
    flat.roofline = roof(10.0)
    flat.rejected = "no delivery scope"
    assert apply_floor([flat]) == []
    assert flat.rejected == "no delivery scope"


def test_frontier_defers_overlapping_alternatives_and_unlocks_shorter_chain():
    full = region("full", 0, 3)
    short = region("short", 0, 1)
    suffix = region("suffix", 2, 3)
    separate = region("separate", 4, 4)
    regions = [short, suffix, separate, full]
    estimates = {
        "full": RegionEstimate(0.50, 0.30),
        "short": RegionEstimate(0.25, 0.20),
        "suffix": RegionEstimate(0.25, 0.10),
        "separate": RegionEstimate(0.20, 0.0),
    }

    ready, deferred = select_frontier(regions, estimates)
    assert [r.fingerprint for r in ready] == ["full", "separate"]
    assert deferred == {"short": ["full"], "suffix": ["full"]}
    assert all(r.rejected is None for r in regions)
    assert {r.fingerprint for r in ready} | set(deferred) == {r.fingerprint for r in regions}

    # The longer cut failed, so both shorter alternatives get a turn.
    remaining = [r for r in regions if r.fingerprint in deferred]
    next_ready, next_deferred = select_frontier(remaining, estimates)
    assert [r.fingerprint for r in next_ready] == ["short", "suffix"]
    assert next_deferred == {}


def test_frontier_can_choose_shorter_region_first_and_records_all_conflicts():
    full = region("full", 0, 3)
    left, right = region("left", 0, 1), region("right", 2, 3)
    estimates = {"left": RegionEstimate(0.3, 0.2), "right": RegionEstimate(0.3, 0.2),
                 "full": RegionEstimate(0.5, 0.1)}
    ready, deferred = select_frontier([full, right, left], estimates)
    assert [r.fingerprint for r in ready] == ["left", "right"]
    assert deferred == {"full": ["left", "right"]}


def test_frontier_preserves_zero_estimates_and_separate_workloads():
    a, b = region("a", 0, 1, "prefill"), region("b", 0, 1, "decode")
    ready, deferred = select_frontier([b, a], {})
    assert [r.fingerprint for r in ready] == ["a", "b"]
    assert deferred == {}


def test_all_alternatives_eventually_return_without_a_ship():
    pending = [region(f"prefix_{end}", 0, end) for end in range(12)]
    expected = {r.fingerprint for r in pending}
    seen = []
    while pending:
        ready, deferred = select_frontier(pending, {})
        assert ready
        seen.extend(r.fingerprint for r in ready)
        pending = [r for r in pending if r.fingerprint in deferred]
    assert len(seen) == len(expected)
    assert set(seen) == expected


def chain_trace(ops=("mx.exp", "mx.exp")):
    spec = ((1024 * 1024,), "float32")
    nodes = tuple(TraceNode(i, op, (i,), (i + 1,), (spec,), (spec,),
                            {"args": (), "kwargs": {}}, "@0", i, ("@0",))
                  for i, op in enumerate(ops))
    return Trace(nodes, {}, (len(nodes),), frozenset(), frozenset({0}), {})


def test_static_estimate_counts_deleted_traffic_and_copies_without_pricing():
    trace = chain_trace()
    singleton, chain = region("single", 0, 0), region("chain", 0, 1)
    peaks = Peaks(100.0, {"float32": 2000.0}, 4.0)
    estimates = estimate_regions([singleton, chain], {"w": trace}, peaks, {"w": 1.0})
    assert estimates["single"].removable_p == pytest.approx(0.0)
    assert estimates["chain"].removable_p == pytest.approx(2 * 1024 * 1024 * 4 / 100e9 * 1000)
    assert estimates["chain"].combined_p == pytest.approx(2 * estimates["single"].combined_p)
    assert chain.p == {} and chain.roofline is None

    chain.members.append(Stretch("other", 0, 1, (0,), (2,), ("@0",)))
    repeated = estimate_regions([chain], {"w": trace, "other": trace}, peaks,
                                {"w": 1.0, "other": 2.0})
    assert repeated["chain"].removable_p == pytest.approx(1.5 * estimates["chain"].removable_p)


def test_static_estimate_views_have_no_launch_or_memory_cost():
    trace = chain_trace(("mx.reshape", "mx.exp"))
    singleton, with_view = region("single", 1, 1), region("view", 0, 1)
    estimates = estimate_regions([singleton, with_view], {"w": trace},
                                 Peaks(100.0, {"float32": 2000.0}, 4.0), {"w": 1.0})
    assert estimates["single"].combined_p == pytest.approx(estimates["view"].combined_p)
    assert estimates["view"].removable_p == pytest.approx(0.0)


def test_missing_step_estimate_cannot_reject_anything():
    candidate = region("unknown", 0, 1)
    estimates = estimate_regions([candidate], {"w": chain_trace()},
                                 Peaks(100.0, {"float32": 2000.0}, 4.0), {})
    assert estimates["unknown"] == RegionEstimate(0.0, 0.0)
    assert select_frontier([candidate], estimates) == ([candidate], {})
    assert candidate.rejected is None


def test_group_price_centers_drift_before_dividing_and_subtracting_link():
    from statistics import median
    from autotuner.regions.price import _group_price

    # A symmetric block has a 100 ms model, 24 ms library loop, 4 ms link,
    # and 10 ms probe at its midpoint. Model endpoints differ by 40%, while
    # the nearer library/link/probe observations differ less. The next block
    # runs 20% slower overall, with exactly the same true shares/headroom.
    def observations(cost, offset):
        return tuple(cost * center * scale for center in (1.0, 1.2)
                     for scale in (1 - offset, 1 + offset))

    rows = {"step": observations(100, 0.2), "r:library": observations(24, 0.05),
            "r:link": observations(4, 0.025), "r:probe": observations(10, 0.01)}
    price = _group_price(rows, "r", iters=10, linked=True)
    assert price.share == pytest.approx(0.02)
    assert price.ms == pytest.approx(2.2)
    assert price.floor_ms == pytest.approx(0.66)
    assert price.stability == pytest.approx(1.0)
    old_share = median((a - b) / 10 / step for a, b, step in
                       zip(rows["r:library"], rows["r:link"], rows["step"]))
    assert old_share > price.share * 1.03  # row-wise endpoint ratios overstate the same region


def test_group_price_without_a_chain_link():
    from autotuner.regions.price import _group_price

    rows = {"step": (100.0, 100.0), "r:library": (20.0, 20.0), "r:probe": (5.0, 5.0)}
    price = _group_price(rows, "r", iters=2, linked=False)
    assert price.share == pytest.approx(0.10)
    assert price.floor_ms == pytest.approx(2.5)


def heterogeneous_region():
    nodes, members = [], []
    for seq, size in enumerate((8, 16, 8)):
        spec = ((size,), "float32")
        nodes.append(TraceNode(seq, "mx.exp", (2 * seq,), (2 * seq + 1,),
                               (spec,), (spec,), {"args": (), "kwargs": {}}, "@0", seq, ("@0",)))
        members.append(Stretch("w", seq, seq, (2 * seq,), (2 * seq + 1,), ("@0",)))
    return (Region("heterogeneous", ("mx.exp",), members),
            Trace(tuple(nodes), {}, (5,), frozenset(), frozenset(), {}))


def test_capture_instances_separates_shapes_but_groups_identical_copies():
    from autotuner.regions.price import capture_instances

    candidate, trace = heterogeneous_region()
    instances = capture_instances(candidate, {"w": trace})
    assert [(label, member.start_seq, copies) for label, member, copies in instances] == \
        [("w", 0, 2), ("w@copy:1", 1, 1)]
    assert candidate.copies == 3


def test_group_pricing_sums_actual_shape_prices(monkeypatch):
    from autotuner.regions import price as pricing

    candidate, trace = heterogeneous_region()

    class Store:
        def set_count(self, fingerprint, label):
            return 1

        def load(self, fingerprint, label, index, kind):
            return {"label": label}

    class Session:
        def log(self, *args, **kwargs):
            pass

        def settle(self):
            pass

    prepared = []

    def replay(session, trace, member, sets, weights, target, baseline):
        prepared.append(member.start_seq)
        return lambda: None, 1, None, sets

    def sample(session, arms, pairs):
        rows = {"step": (100.0,) * pairs}
        for key in arms:
            if key == "step":
                continue
            cost = 4.0 if "@copy:1" in key else 1.0
            rows[key] = ((cost / 2 if key.endswith(":probe") else cost),) * pairs
        return rows

    monkeypatch.setattr(pricing, "_looped_replay", replay)
    monkeypatch.setattr(pricing, "_probe_pass", lambda *args: lambda: None)
    monkeypatch.setattr(pricing, "chained_loop", lambda *args: lambda: None)
    monkeypatch.setattr(pricing, "sample_group", sample)
    pricing.price_group([candidate], Session(), {"w": trace}, Store(), {"w": lambda: None}, pairs=4)

    assert prepared == [0, 1]
    assert candidate.t_rep_ms["w"] == 1.0
    assert candidate.t_orig_ms["w"] == 6.0  # two 1 ms copies plus one 4 ms copy
    assert candidate.p["w"] == pytest.approx(0.06)
    assert candidate.prices["w@copy:1"].ms == 4.0
    assert candidate.prices["w@copy:1"].floor_ms == 2.0
