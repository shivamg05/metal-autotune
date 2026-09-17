"""Failures observed during the FLUX preparation and delivery review."""

import json
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner.e2e import preserving_check, run_e2e
from autotuner.measure.clocks import loop_iterations, sample_group
from autotuner.measure.session import Session


@pytest.mark.parametrize("bad", [
    lambda x: x * float("nan"),
    lambda x: x[:, :1],
    lambda x: x.astype(mx.float16),
    lambda x: [],
    lambda x: [x],
])
def test_model_check_rejects_corrupt_output_contract(bad):
    x = mx.ones((2, 3))
    assert not preserving_check(lambda: x, lambda: bad(x), "w").passed


def test_model_check_keeps_small_outputs_independent_of_large_ones():
    reference = [mx.array([1e6]), mx.array([1.0])]
    corrupted = [reference[0], mx.array([2.0])]
    assert not preserving_check(lambda: reference, lambda: corrupted, "w").passed


def test_model_check_accepts_matching_nonfinite_patterns():
    x = mx.array([float("nan"), float("inf"), -float("inf"), 1.0])
    assert preserving_check(lambda: x, lambda: mx.array(x), "w").passed


def test_slow_region_does_not_force_twenty_repeats():
    passes = []

    def loop_for(n):
        def run():
            passes.append(n)
            return 0.2 * n + 0.001
        return run

    assert loop_iterations(lambda fn: fn(), loop_for) == 1
    assert sum(passes) < 10


def test_loop_sizing_subtracts_fixed_submission_cost():
    def size(overhead):
        return loop_iterations(lambda fn: fn(), lambda n: lambda: overhead + n * 0.00001)
    assert size(0.0001) == pytest.approx(size(0.010), abs=1)


def test_group_shares_model_arm_and_reverses_order():
    order = []

    class FakeSession:
        def fresh_chunk(self, fn): pass
        def warm_until_stable(self, fn): pass
        def timed(self, fn): return fn()
        def settle(self): pass
        def log(self, *args, **kwargs): pass

    def arm(name, cost):
        def run():
            order.append(name)
            return cost
        return run

    samples = sample_group(FakeSession(), {
        "model": arm("model", 1.5), "a": arm("a", 0.01), "b": arm("b", 0.02)}, pairs=4)
    assert order == ["model", "a", "b", "b", "a", "model"] * 2
    assert samples["model"] == (1500.0,) * 4


def test_cooling_is_logged_before_sleep(tmp_path):
    path = tmp_path / "session.jsonl"
    seen = []
    def sleep(_seconds):
        seen.append(json.loads(path.read_text().splitlines()[-1])["kind"])
    session = Session(log_path=path, sleep=sleep)
    session._debt_s = 2.0
    session.settle()
    assert seen == ["cooling"]
    assert session.idled_s == 6.0


def test_failed_artifact_validation_preserves_existing_artifact(tmp_path):
    from autotuner.artifact.emit import emit_artifact
    from autotuner.report import Report
    target = tmp_path / "artifact"
    target.mkdir()
    (target / "keep.txt").write_text("previous verified result")
    def fail(_staged):
        raise RuntimeError("fresh-process output differs")
    with pytest.raises(RuntimeError, match="output differs"):
        emit_artifact(target, [], [], Report(manifest_path="fixture"), validate=fail)
    assert (target / "keep.txt").read_text() == "previous verified result"
    assert not target.with_name("artifact.building").exists()


def test_mixed_dtype_compute_estimate_uses_each_ops_throughput():
    from autotuner.measure.peaks import Peaks
    from autotuner.regions.roofline import compute_time_ms
    from autotuner.trace.types import TraceNode
    def op(seq, dtype, count):
        return TraceNode(seq, "mx.add", (0, 1), (2,), (),
                         (((count,), dtype),), {}, "@0", seq)
    nodes = [op(0, "float32", 1), op(1, "bfloat16", 1000000)]
    peaks = Peaks(100, {"float32": 1000, "bfloat16": 2000})
    assert compute_time_ms(nodes, peaks) == pytest.approx(1e-9 + 0.0005)


def test_e2e_times_every_declared_workload(monkeypatch):
    from autotuner import e2e
    from types import SimpleNamespace
    timed_sizes = []
    def clock(session, baseline, candidate, pairs, **kwargs):
        timed_sizes.append(baseline().size)
        return SimpleNamespace(), True
    monkeypatch.setattr(e2e, "step_veto", clock)
    result = run_e2e(Session(), lambda x: x, lambda x: mx.array(x),
                     [("small", [mx.ones(2)]), ("large", [mx.ones(5)])])
    assert result.passed
    assert timed_sizes == [2, 5]
    assert set(result.workload_vetos) == {"small", "large"}


def test_failed_preparation_leaves_report_and_restores_tracer(tmp_path, monkeypatch):
    from autotuner.loop import JobRunner
    manifest = tmp_path / "manifest.yaml"
    fixture = Path(__file__).parent / "fixtures" / "planted_win.py"
    manifest.write_text(f"model: {fixture}\nworkloads:\n  - inputs: [{{shape: [4, 1024], dtype: float32}}]\n")
    runner = JobRunner(manifest, tmp_path / "work", lambda r: None)
    def broken():
        runner.tracer.install()
        raise RuntimeError("preparation failed")
    monkeypatch.setattr(runner, "load_model", broken)
    with pytest.raises(RuntimeError, match="preparation failed"):
        runner.run()
    report = json.loads((tmp_path / "work" / "report.json").read_text())
    assert report["session"]["status"] == "failed"
    assert report["session"]["stage"] == "loading model"
    assert runner.tracer.verify_restored() == []


def test_ship_compares_with_current_incumbent_on_every_workload():
    from autotuner.loop import JobRunner
    from autotuner.e2e import E2EResult, WorkloadCheck
    from autotuner.measure.clocks import comparison_from_samples
    runner = object.__new__(JobRunner)
    runner.model_ratio = 0.50  # an old ratio from another measurement window
    win = comparison_from_samples([100.0] * 8, [98.0] * 8)
    loss = comparison_from_samples([100.0] * 8, [101.0] * 8)
    check = WorkloadCheck("w", 0, 0, 0, 1, True)
    result = E2EResult([check], win, True, {"first": win, "second": win})
    assert runner._model_win(result)
    result.workload_vetos["second"] = loss
    assert not runner._model_win(result)


def test_region_clock_must_nominate_before_model_is_built(monkeypatch):
    from types import SimpleNamespace
    from autotuner.loop import JobRunner
    runner = object.__new__(JobRunner)
    def forbidden():
        raise AssertionError("a slower region should not pay for model validation")
    monkeypatch.setattr(runner, "_copy_incumbent", forbidden)
    assert not runner._bind_and_promote(None, None, SimpleNamespace(outcome="correct_slower"))


def test_symmetric_clock_subtracts_link_in_the_same_window():
    from autotuner.measure.clocks import paired_means, comparison_from_samples
    # Four arms move together under linear drift. Their centered differences
    # must retain the true 2 ms win after removing the shared chain cost.
    rows = {name: [] for name in ("library", "candidate", "probe", "link")}
    costs = {"library": 12.0, "candidate": 10.0, "probe": 6.0, "link": 2.0}
    order = list(rows)
    for _ in range(8):
        for name in order:
            rows[name].append(costs[name] + sum(map(len, rows.values())) * 0.05)
        order.reverse()
    centered = {k: paired_means(v) for k, v in rows.items()}
    net = {k: [v - l for v, l in zip(centered[k], centered["link"])]
           for k in ("library", "candidate")}
    clock = comparison_from_samples(net["library"], net["candidate"])
    assert clock.median_baseline_ms == pytest.approx(10)
    assert clock.median_delta_ms == pytest.approx(2)
    assert clock.wins_by(0.1)


def test_buffer_limit_counts_scratch_and_mlx_metadata():
    from autotuner.ladder.static_checks import buffer_count
    from autotuner_runtime.kernels import KernelSpec
    spec = KernelSpec("many", "many", ("in0", "in1"), tuple(f"tmp{i}" for i in range(28)),
                      "// in0_shape in0_strides in1_ndim")
    # MLX scans the source text, including comments. Scalar arrays receive no
    # shape/stride/ndim buffers even when those strings appear.
    assert buffer_count(spec, (2, 0)) == 32


def test_scaffold_repair_counts_toward_the_job_budget(monkeypatch):
    from types import SimpleNamespace
    from autotuner.loop import JobRunner, RegionRun
    runner = object.__new__(JobRunner)
    runner.manifest = SimpleNamespace(budget_per_region=2, budget_total=1)
    runner._finish_requested = lambda: False
    runner.total_hypotheses = 0
    runner.log = SimpleNamespace(append=lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(runner, "_meta", lambda *args: {})
    monkeypatch.setattr(runner, "_note_lesson", lambda *args: None)
    monkeypatch.setattr(runner, "_ask_judge", lambda *args: (calls.append(1) or SimpleNamespace(kernel=None), None))
    run = RegionRun(SimpleNamespace(fingerprint="region", roofline=None))
    kernel = SimpleNamespace(kernel_id="scaffold")
    result = SimpleNamespace(failed_gate="compile", detail={}, region_ms=None, library_ms=None,
                             win_ms=None, sigma_ms=None, floor_ms=None)
    assert runner._judge_fix(run, None, kernel, result) is None
    assert (runner.total_hypotheses, run.hypotheses) == (1, 1)
    assert runner._judge_fix(run, None, kernel, result) is None
    assert calls == [1]


def test_a_replay_guard_treats_a_recorded_none_with_is(tmp_path):
    """work-2026-09-16-1332 crashed on its first region: a recurrent block
    takes cache=None on the prompt call and an array on the next, and the
    wrapper's guard compared with ==, which mlx refuses for array vs None.
    The guard must use `is None` and hand the array call to the original."""
    from types import SimpleNamespace
    from autotuner.loop import JobRunner, _load_class, _resolve
    from autotuner_runtime.state import correctness_call
    from autotuner_runtime.swap import install as swap_install

    model_file = tmp_path / "tiny_recurrent_gemma.py"
    model_file.write_text(
        "def build():\n"
        "    from mlx_lm.models.recurrent_gemma import Model, ModelArgs\n"
        "    return Model(ModelArgs(model_type='recurrent_gemma', attention_bias=False, conv1d_width=4,\n"
        "                           hidden_size=64, intermediate_size=128, logits_soft_cap=30.,\n"
        "                           num_attention_heads=4, num_hidden_layers=2, num_key_value_heads=1,\n"
        "                           rms_norm_eps=1e-6, rope_theta=10000., attention_window_size=64,\n"
        "                           vocab_size=64, block_types=['recurrent', 'attention']))\n")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"model: {model_file}\nworkloads:\n  - name: prefill\n    inputs:\n"
                        "      - shape: [1, 12]\n        dtype: int32\n        low: 0\n        high: 64\n"
                        "budget: {per_region: 1, total: 1}\nfinal_benchmark: {steps: 1, pairs: 4, warmup_steps: 1}\n")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda _: None, session=Session())
    runner.load_model()
    runner.trace_workloads()
    runner.tracer.uninstall()
    trace = runner.traces["prefill"]
    scope = next(sc for sc in trace.scope_calls if sc.address == "model.model.layers.0.temporal_block.rg_lru@0")
    emitted = runner._emit_installation([(trace, scope, [])], "Id_rg_lru", replay=True)
    assert "cache is None" in emitted.source and "cache == None" not in emitted.source
    path = "model.model.layers.0.temporal_block.rg_lru"
    swap_install(runner.baseline_model, path, _load_class(emitted)(_resolve(runner.baseline_model, path), {}))
    # the prompt call replays; the generated token's call carries an array cache and runs the original
    correctness_call(runner.baseline_model, runner.tensors["prefill"])


def test_certification_blames_the_scope_that_ran_first():
    """work-2026-09-16-1332 spent 47 minutes settling delivery: one scope whose
    compiled output differed changed every scope after it, and the batch check
    blamed whichever mismatch sorted first by name, one per full-model pass,
    143 times. The earliest-executed mismatch is the one to blame."""
    import mlx.nn as nn
    from autotuner.bind.certify import certify_identities

    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.z_first = nn.Linear(4, 4)
            self.a_second = nn.Linear(4, 4)

        def __call__(self, x):
            return self.a_second(self.z_first(x))

    class Exact(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def __call__(self, x):
            return self.wrapped(x)

    class Off(Exact):
        def __call__(self, x):
            return self.wrapped(x) + 1

    model = Two()
    x = mx.ones((2, 4))
    wrappers = {"a_second": Exact(model.a_second), "z_first": Off(model.z_first)}  # name order blames a_second
    check = certify_identities(model, wrappers, [lambda: model(x)], repeated_calls=1)
    assert not check.ok and check.scope == "z_first", (check.scope, check.reason)


def test_a_trace_pins_no_array_versions():
    """work-2026-09-16-1332 traced a 500-token RecurrentGemma prefill in 109 s
    at a 59 GB peak: the recorder kept a value snapshot of every version of
    the scan's 5 MB output, 9,000 of them, and the library's mid-pass eval
    made them all resident. A trace needs shapes and ids only; a capture
    keeps snapshots for the boundary it came for and nothing else."""
    import mlx.nn as nn
    from autotuner.trace import Tracer
    from autotuner.regions.price import capture_boundaries

    class Scan(nn.Module):
        def __call__(self, x):
            y = mx.zeros_like(x)                 # 4 MB, rewritten in place per step
            h = mx.zeros((x.shape[2],))
            for t in range(x.shape[1]):
                h = x[:, t] * 0.5 + h
                y[:, t] = h
            mx.eval(y)                           # the library evaluates mid-pass
            return y * 2

    model = Scan()
    x = mx.ones((1, 200, 5000))
    tracer = Tracer()
    tracer.install()
    try:
        mx.eval(model(x))
        mx.reset_peak_memory()
        trace, _ = tracer.trace(model, [x])
        peak = mx.get_peak_memory()
        assert not tracer.recorder._by_aid and tracer.recorder.snapshot_seqs is None
        # a handful of 4 MB versions in flight, never two hundred
        assert peak < 20 * x.nbytes, f"trace peak {peak/1e6:.0f} MB for a {x.nbytes/1e6:.0f} MB array"
        setitems = [n for n in trace.nodes if n.op.endswith("__setitem__")]
        assert len(setitems) == 200
        # a capture keeps exactly the boundary it came for
        wanted = set(trace.nodes[-1].out_arrays) | set(setitems[3].out_arrays)
        got = capture_boundaries(tracer, model, [x], trace, wanted)
        assert set(got) == wanted and tracer.recorder.snapshot_seqs is None
    finally:
        tracer.uninstall()
