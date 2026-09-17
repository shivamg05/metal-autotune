"""MetalBench problems become ordinary jobs: the bridge reads the vendored
registries, writes a manifest the loader accepts and a model that runs at the
registered shapes; the scorer reads a report into the two baselines."""

import mlx.core as mx
import pytest

from autotuner import manifest as manifest_mod
from metalbench.bridge import problems, write_job
from metalbench.run import fast_p, score


def test_every_registered_problem_has_a_module_and_shapes():
    found = problems()
    assert len(found) >= 100
    assert all(p.module.exists() and p.input_shapes for p in found.values())
    assert {p.set for p in found.values()} == {"common", "standard", "full"}


@pytest.mark.parametrize("name", ["abs", "rms_norm_linear", "transformer_block"])
def test_a_problem_becomes_a_job_that_builds_and_runs(tmp_path, name):
    problem = problems()[name]
    manifest = write_job(problem, budget_per_region=2, budget_total=2, out=tmp_path)
    loaded = manifest_mod.load(manifest)
    assert [tuple(int(d) for d in i.shape) for i in loaded.workloads[0].inputs] == list(problem.input_shapes)
    assert loaded.tolerances is not None and loaded.baseline == "compiled"
    import importlib.util
    spec = importlib.util.spec_from_file_location("job_model", manifest.parent / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built = module.build()
    inputs = [mx.random.normal(shape) for shape in problem.input_shapes]
    out = built(*inputs)
    mx.eval(out)
    assert isinstance(out, mx.array)


def test_scores_read_both_baselines_and_count_strict_wins():
    report = {"session": {"status": "ok"},
              "baseline": {"choice": "compiled", "clocks_ms": {"p": {"plain": 3.0, "compiled": 2.0}}},
              "step_ms": {"p": {"speedup": 1.25, "win_confirmed": True}},
              "regions": [{"s": 1.3}, {}]}
    row = score(report, "p")
    assert row["vs_compiled"] == 1.25 and row["vs_eager"] == pytest.approx(1.875) and row["shipped_regions"] == 1
    nothing = score({"session": {"status": "ok"}, "baseline": {"choice": "compiled", "clocks_ms": {"p": {"plain": 3.0, "compiled": 2.0}}},
                     "step_ms": {"p": {"speedup": 1.4, "win_confirmed": False}}}, "p")
    assert nothing["vs_compiled"] == 1.0 and nothing["vs_eager"] == pytest.approx(1.5)
    failed = score({"session": {"status": "failed"}}, "p")
    assert failed["vs_eager"] == 1.0 and failed["vs_compiled"] is None
    rows = [row, nothing, {"vs_compiled": 1.0, "vs_eager": 1.0}]
    fp = fast_p(rows, "vs_compiled")
    assert fp["fast_1"] == pytest.approx(1 / 3) and fp["fast_1.25"] == 0.0
    assert fast_p(rows, "vs_eager")["fast_1.25"] == pytest.approx(2 / 3)


def test_a_problem_runs_inside_a_swappable_child_scope(tmp_path):
    """The first end-to-end run stranded all 11 regions of rms_norm_linear as
    'runs at the model's top level': the wrapper called forward() directly.
    The vendored forward must run inside a child module's call so a region
    there has a scope a wrapper can replace."""
    import importlib.util
    from autotuner.trace import Tracer

    manifest = write_job(problems()["rms_norm_linear"], budget_per_region=1, budget_total=1, out=tmp_path)
    spec = importlib.util.spec_from_file_location("job_model_scope", manifest.parent / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built = module.build()
    inputs = [mx.random.normal(shape) for shape in problems()["rms_norm_linear"].input_shapes]
    tracer = Tracer()
    tracer.install()
    try:
        trace, _ = tracer.trace(built, inputs)
    finally:
        tracer.uninstall()
    assert any(sc.address == "model@0" for sc in trace.scope_calls)
    assert all(n.module_address == "model@0" for n in trace.nodes)
