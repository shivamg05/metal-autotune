"""The measurement kit. Timing tests use real GPU work and modest sample
counts; assertions are on paired quantities so machine load cannot flip them.
"""

import mlx.core as mx
import pytest

from autotuner.measure.clocks import compare, step_clock
from autotuner.measure.controls import aa_null, synthetic_workload
from autotuner.measure.peaks import measure_bandwidth, measure_flops, measure_launch_us
from autotuner.measure.session import Session, time_once


@pytest.fixture(scope="module")
def session():
    return Session()


def require_quiet_machine():
    """Some assertions here are about chip physics and the session floor; a
    machine busy with other work cannot measure either. The
    paired design survives load, but a 3% planted signal does not clear a
    contention-inflated floor, and a starved bandwidth run is not a peak.
    Thermal throttle is the same problem at zero load, so both gates apply."""
    from tests.conftest import require_healthy_gpu, require_quiet_load

    require_quiet_load()
    require_healthy_gpu()


def quiet_session():
    """No real idling, for tests that check mechanics rather than numbers."""
    return Session(sleep=lambda s: None)


def test_time_once_scales_with_work():
    # depth ratios only hold while per-launch overhead is small next to the
    # work; on a throttled chip the overhead dominates and 8x depth reads 2.8x
    require_quiet_machine()
    small = synthetic_workload(depth=2)
    big = synthetic_workload(depth=16)
    s = quiet_session()
    s.warm_until_stable(small)
    s.warm_until_stable(big)
    t_small = min(time_once(small) for _ in range(3))
    t_big = min(time_once(big) for _ in range(3))
    assert t_big > 3 * t_small


def test_session_settle_pays_debt():
    slept = []
    s = Session(sleep=slept.append, duty_idle_factor=3.0)
    fn = synthetic_workload(depth=2)
    t = s.timed(fn)
    s.settle()
    assert len(slept) == 1
    assert slept[0] == pytest.approx(3.0 * t, rel=1e-6)
    s.settle()
    assert len(slept) == 1  # debt already paid


def test_fresh_chunk_paces_and_ramp_warms_past_threshold():
    slept = []
    s = Session(sleep=slept.append, max_chunk_work_s=0.0)
    fn = synthetic_workload(depth=2)
    calls = {"n": 0}

    def counted():
        calls["n"] += 1
        return fn()

    s.timed(counted)
    s.fresh_chunk((counted,))
    assert len(slept) == 1  # debt past a zero threshold pays immediately
    assert calls["n"] == 3  # 1 timed + 2 unmeasured ramp-warm samples


def test_fresh_chunk_noop_below_threshold():
    slept = []
    s = Session(sleep=slept.append, max_chunk_work_s=1e9)
    fn = synthetic_workload(depth=2)
    s.timed(fn)
    s.fresh_chunk((fn,))
    assert slept == []


def test_warm_until_stable_converges(session):
    times = session.warm_until_stable(synthetic_workload(depth=4))
    assert 2 <= len(times) <= 30
    converged = abs(times[-1] - times[-2]) <= max(0.01 * times[-2], 100e-6)
    assert converged or len(times) == 30


def test_step_clock_reports_median(session):
    clock = step_clock(session, synthetic_workload(depth=4), reps=5)
    assert clock.median_ms > 0
    assert len(clock.samples_ms) == 5
    assert min(clock.samples_ms) <= clock.median_ms <= max(clock.samples_ms)


def test_aa_null_shows_no_significant_win(session):
    """Law 12: the A/A control must not ship. Same fn both arms."""
    result = aa_null(session, pairs=12)
    assert not result.wins_by(margin_ms=0.0)
    assert not result.loses_by(margin_ms=0.0)


def test_injected_slowdown_is_bit_identical():
    base = synthetic_workload(depth=8, seed=3)
    injected = synthetic_workload(depth=8, injected_links=2, seed=3)
    a, b = base(), injected()
    mx.eval(a, b)
    assert mx.array_equal(a, b).item()


def test_injected_slowdown_is_detected(session):
    """A known ~3% planted slowdown is caught by the paired
    comparison while outputs stay bit-identical."""
    require_quiet_machine()
    base = synthetic_workload(depth=33, seed=5)
    injected = synthetic_workload(depth=33, injected_links=1, seed=5)
    result = compare(session, base, injected, pairs=16)
    assert result.loses_by(margin_ms=0.0), (
        f"planted 3% slowdown not detected: delta={result.median_delta_ms:.4f}ms "
        f"sigma={result.sigma_ms:.4f}ms base={result.median_baseline_ms:.2f}ms"
    )
    slow_pct = -result.median_delta_ms / result.median_baseline_ms * 100
    assert 0.5 < slow_pct < 8.0, f"implausible slowdown size {slow_pct:.2f}%"


def test_compare_rejects_odd_pairs(session):
    with pytest.raises(ValueError):
        compare(session, lambda: mx.zeros(1), lambda: mx.zeros(1), pairs=3)


def test_peaks_are_physically_sane(session):
    require_quiet_machine()
    bw = measure_bandwidth(session, samples=3)
    assert 30 < bw < 1000, f"bandwidth {bw:.0f} GB/s out of any plausible M-series range"
    fp32 = measure_flops(session, mx.float32, samples=3)
    fp16 = measure_flops(session, mx.float16, samples=3)
    assert 500 < fp32 < 50_000, f"fp32 {fp32:.0f} GFLOPs implausible"
    assert fp16 > 0.5 * fp32, "fp16 should not be far below fp32 on Apple GPUs"
    launch = measure_launch_us(session, samples=3)
    assert 0.1 < launch < 100, f"launch cost {launch:.2f}us implausible"


def test_implausible_peaks_are_named():
    """The degraded-machine guard (seen live: a GPU at a tenth of its known
    speed within one boot session)."""
    from autotuner.measure.peaks import Peaks, implausible

    healthy = Peaks(bandwidth_gbps=92.8, flops_gflops={"float32": 2800.0}, launch_us=6.0)
    assert implausible(healthy) is None
    slow_bw = Peaks(bandwidth_gbps=9.4, flops_gflops={"float32": 2800.0})
    assert "bandwidth" in implausible(slow_bw)
    slow_fp = Peaks(bandwidth_gbps=92.8, flops_gflops={"float32": 180.0})
    assert "fp32" in implausible(slow_fp)
    no_fp32 = Peaks(bandwidth_gbps=92.8, flops_gflops={})
    assert implausible(no_fp32) is None


def test_fresh_chunk_ramp_warm_stops_once_enough_work_ran():
    """After an idle the clocks ramp back up within a few tens of ms of work;
    a sample longer than that warms them by itself, so a slow model step must
    not pay two unmeasured full steps per timed one."""
    import time as _time

    s = Session(sleep=lambda x: None, max_chunk_work_s=0.0)
    calls = {"n": 0}

    def slow():
        calls["n"] += 1
        _time.sleep(0.06)
        return mx.zeros(1)

    s.timed(slow)
    s.fresh_chunk((slow,))
    assert calls["n"] == 2  # one timed sample, one ramp sample


def test_best_of_peaks_keeps_the_higher_reading():
    from autotuner.measure.peaks import Peaks, best_of

    a = Peaks(90.0, {"float32": 2000.0, "float16": 3000.0}, 7.0)
    b = Peaks(95.0, {"float32": 1800.0}, 6.0)
    m = best_of(a, b)
    assert (m.bandwidth_gbps, m.launch_us) == (95.0, 6.0)
    assert m.flops_gflops == {"float32": 2000.0, "float16": 3000.0}


def test_gpu_utilization_parses_what_macos_reports():
    from autotuner.measure.peaks import gpu_utilization

    text = '"PerformanceStatistics" = {"Tiler Utilization %"=98,"Device Utilization %"=37,"x"=1}'
    assert gpu_utilization(text) == 37.0
    assert gpu_utilization("nothing about the GPU here") is None


def test_gpu_core_count_parses_what_macos_reports():
    from autotuner.measure.peaks import gpu_core_count

    text = 'some other line\n      "gpu-core-count" = 10\n      "IOClass" = "AGXAcceleratorG16X"\n'
    assert gpu_core_count(text) == 10
    assert gpu_core_count("nothing here") is None


def test_chained_loop_computes_the_same_values_and_prices_its_link():
    """The chain adds a zero from the last pass to the next pass's linked
    input, so every pass computes exactly what the unchained loop computes,
    the link rides on the smallest non-weight float input, and the link
    loop alone hands every input back unchanged."""
    from autotuner.measure.clocks import chained_loop, link_input, link_loop, timing_sets
    from autotuner.measure.session import time_once

    w = mx.random.normal((256, 64), key=mx.random.key(1))
    x = mx.random.normal((4, 64), key=mx.random.key(2))
    mx.eval(w, x)
    sets = timing_sets([{7: w, 3: x}])
    assert len(sets) > 1 and all(set(s) == {3, 7} for s in sets)
    assert link_input(sets[0], weight_ids={7}) == 3
    assert link_input({7: w, 3: x}) == 3  # the smallest float input even with no weight ids
    assert link_input({1: mx.zeros((4,), dtype=mx.int32)}) is None

    passes = lambda b: [b[3] @ b[7].T]
    chained = chained_loop(passes, sets, 12, 3)()
    plain = [passes(sets[i % len(sets)]) for i in range(12)]
    mx.eval(chained, plain)
    assert all(mx.array_equal(c[0], p[0]).item() for c, p in zip(chained, plain))
    linked = link_loop(sets, 12, 3)()
    mx.eval(linked)
    assert all(mx.array_equal(l[0], sets[i % len(sets)][3]).item() for i, l in enumerate(linked))


def test_chained_launches_do_not_overlap():
    """Protects: the region clock's chain law. Metal runs independent launches
    side by side, so a one-threadgroup kernel timed in an unchained loop reads
    several times faster than it runs in a model, where each layer waits for
    the last (seen live on the 2026-09-02 Qwen run: 0.03 ms unchained, 0.33
    chained, 0.36 in the model)."""
    from tests.conftest import require_healthy_gpu, require_quiet_load
    from autotuner.measure.clocks import chained_loop
    from autotuner.measure.session import time_once

    require_healthy_gpu()
    require_quiet_load()
    kernel = mx.fast.metal_kernel(
        name="pin_one_threadgroup_matvec", input_names=["w", "x"], output_names=["out"],
        source="""
            uint r = thread_position_in_grid.x;
            float acc = 0.0f;
            for (uint c = 0; c < 2048u; ++c) { acc += (float)w[r * 2048u + c] * (float)x[c]; }
            out[r] = acc;
        """)
    ws = [mx.random.normal((128, 2048), key=mx.random.key(i)).astype(mx.float16) for i in range(8)]
    x = mx.random.normal((2048,), key=mx.random.key(99)).astype(mx.float16)
    mx.eval(ws, x)
    sets = [{0: w, 1: x} for w in ws]
    one = lambda b: kernel(inputs=[b[0], b[1]], grid=(128, 1, 1), threadgroup=(128, 1, 1),
                           output_shapes=[(128,)], output_dtypes=[mx.float32])
    n = 40
    unchained = lambda: [one(sets[i % len(sets)]) for i in range(n)]
    chained = chained_loop(one, sets, n, 1)
    for _ in range(3):
        time_once(unchained)
        time_once(chained)
    t_un = min(time_once(unchained) for _ in range(5))
    t_ch = min(time_once(chained) for _ in range(5))
    assert t_ch > 2.0 * t_un, (t_un, t_ch)
