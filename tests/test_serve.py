"""manifest serve: after the search, shipped kernels run at every size in a
range they were checked right at, through generated code that reads the size
from its argument; everything else runs the original module."""

import mlx.core as mx
import pytest

from autotuner.loop import JobRunner, RegionRun
from autotuner.bind.sized import NotSizeGeneric, generalize
from autotuner.serve import served_range
from autotuner_runtime import kernels
from tests.test_install import WIN, _runner, elementwise


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Row count L optimized at 64, served over 4..96. Elementwise stand-ins
    never beat the library on a clock, so the grid is told they win and the
    whole-model decision is forced, as the install tests do."""
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, e2e: True)
    monkeypatch.setattr(JobRunner, "_serve_grid", lambda self, job, grid: {n: {"win": True} for n in grid})
    r = _runner(tmp_path, "[L, 1024]", "primary: {L: 64}\n        check_optimizations_for: {L: [4, 96]}")
    yield r
    r.tracer.uninstall()


def _ship(runner, kernel):
    region = runner.region("mx.maximum")
    runner._capture([region])
    assert runner._bind_and_promote(RegionRun(region=region), kernel, WIN)


def _counting(monkeypatch):
    calls = []
    real = kernels.try_sized
    monkeypatch.setattr(kernels, "try_sized",
                        lambda spec, ins, **kw: calls.append(tuple(ins[0].shape)) or real(spec, ins, **kw))
    return calls


def test_a_kernel_runs_at_sizes_the_job_never_recorded(served, monkeypatch):
    _ship(served, elementwise("rmax_s1", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    assert served._serve_sizes()
    report = served.report.serve
    assert report["scopes"]["chain"]["range"] == [4, 96]
    assert report["kernels"]["rmax_s1"]["range"] == [4, 96]
    source = served.emitted["chain"].source
    assert "(ReplayWrapper)" in source
    assert "_kernels.try_sized(_s, _ins) if 4 <= _n <= 96 else None" in source

    calls = _counting(monkeypatch)
    for rows in (37, 5, 96):
        x = mx.random.normal((rows, 1024), key=mx.random.key(rows))
        assert mx.array_equal(served.model(x), served.baseline_model(x)).item()
    assert calls == [(37, 1024), (5, 1024), (96, 1024)]
    calls.clear()
    x = mx.random.normal((100, 1024))  # outside the range: the original module
    assert mx.array_equal(served.model(x), served.baseline_model(x)).item() and calls == []


def test_sizes_where_the_kernel_is_wrong_run_the_original(served, monkeypatch):
    # right from 16 rows up, wrong below: the sandbox check finds each size
    _ship(served, elementwise("rmax_s2", "out0[i] = (in0_shape[0] < 16) ? 1.0f : "
                                         "(in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    assert served._serve_sizes()
    kernel = served.report.serve["kernels"]["rmax_s2"]
    assert kernel["range"] == [16, 96] and kernel["wrong_count"] == 12

    calls = _counting(monkeypatch)
    for rows in (7, 16):
        x = mx.random.normal((rows, 1024), key=mx.random.key(rows))
        assert mx.array_equal(served.model(x), served.baseline_model(x)).item()
    assert calls == [(16, 1024)]


def test_withdrawing_restores_the_searched_wrapper(served):
    _ship(served, elementwise("rmax_s3", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    searched = served.emitted["chain"]
    restore = served._serve_sizes()
    assert served.emitted["chain"] is not searched
    served._unserve(restore, "test")
    assert served.emitted["chain"] is searched and served.report.serve["withdrawn"] == "test"
    assert type(served.model.chain).__name__ == searched.class_name


def test_integers_that_follow_the_size_become_expressions():
    sources = [f"x.reshape(1, {n - 1}, 8)\ny = {n}\nz = 7\n" for n in (10, 20, 35)]
    source, varied = generalize(sources, [10, 20, 35])
    assert source == "x.reshape(1, (_n - 1), 8)\ny = _n\nz = 7\n" and varied == 2
    with pytest.raises(NotSizeGeneric, match="not one integer linear function"):
        generalize(["y = 1\n", "y = 4\n", "y = 9\n"], [1, 2, 3])
    with pytest.raises(NotSizeGeneric, match="different operations"):
        generalize(["y = f(1)\n", "y = f(2)\n", "y = g(3)(4)\n"], [1, 2, 3])


def test_the_served_range_follows_the_timed_wins_and_stops_at_a_loss():
    win, tie, loss = {"win": True}, {}, {"loss": True}
    grid = {100: tie, 400: win, 1000: win, 2048: win}
    assert served_range(list(range(50, 2049)), grid, 2047) == (400, 2048)
    assert served_range(list(range(500, 2049)), grid, 2047) == (1000, 2048)   # 400 is not right
    assert served_range(list(range(1, 2000)), grid, 2047) is None             # not right where it shipped
    # a random sample slower than the library cuts the range; one that ties does not
    samples = {1500: loss, 1600: win, 1200: tie}
    assert served_range(list(range(50, 2049)), {**grid, **samples}, 2047, grid) == (1600, 2048)
    assert served_range(list(range(50, 2049)), {**grid, 1200: tie}, 2047, grid) == (400, 2048)


def test_generated_code_that_computes_something_else_is_never_served(served, monkeypatch):
    """The every-size check compares the generated code's calculation with the
    original module's; one changed constant fails it at every size."""
    import autotuner.serve as serve
    _ship(served, elementwise("rmax_s4", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);"))
    searched = served.emitted["chain"]
    real = serve.plan_scopes

    def tampered(*args, **kwargs):
        plans, refused = real(*args, **kwargs)
        for plan in plans:
            assert "* 2.0)" in plan.sized.template
            plan.sized.template = plan.sized.template.replace("* 2.0)", "* 2.5)", 1)
        return plans, refused

    monkeypatch.setattr(serve, "plan_scopes", tampered)
    assert served._serve_sizes() is None
    scope = served.report.serve["scopes"]["chain"]
    assert scope["range"] is None and scope["first_difference"][1] == "different leaf"  # the 2.5
    assert served.emitted["chain"] is searched


def _prompt_runner(tmp_path, extra=""):
    from pathlib import Path
    from autotuner.measure.session import Session
    model = Path(__file__).parent / "fixtures" / "llama_cache_model.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {model}
baseline: plain
use_library_inference: true
workloads:
  - name: prompt
    context: 0
    inputs: [{{shape: [1, 12], dtype: int32, high: 64}}]
final_benchmark: {{steps: 1, pairs: 4, warmup_steps: 1}}
{extra}""")
    return JobRunner(manifest, tmp_path / "work", judge_factory=lambda _: None,
                     session=Session(sleep=lambda _: None))


def test_an_mlx_lm_prompt_run_checks_every_prompt_length_by_default(tmp_path):
    runner = _prompt_runner(tmp_path)
    try:
        runner.load_model()
        assert dict(runner.manifest.serve) == {"prompt_tokens": (3, 2049)}
        assert runner.manifest.primary["prompt_tokens"] == 12
        runner.trace_workloads()
        # the optimized prompt is unchanged; the ends and middle of the range are recorded too
        assert runner.tensors["prompt"][0].shape == (1, 12)
        assert sorted(t[0].shape[1] for t in runner.serve_tensors.values()) == [3, 1026, 2049]
    finally:
        runner.tracer.uninstall()


def test_check_optimizations_for_false_keeps_the_single_size(tmp_path):
    runner = _prompt_runner(tmp_path, "check_optimizations_for: false\n")
    try:
        runner.load_model()
        runner.trace_workloads()
        assert not runner.manifest.serve and not runner.serve_tensors
    finally:
        runner.tracer.uninstall()
