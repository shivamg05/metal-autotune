"""MetalBench problems become ordinary jobs: the bridge reads the vendored
registries, writes a manifest the loader accepts and a model that runs at the
registered shapes; the scorer reads a report into the two baselines."""

import mlx.core as mx
import pytest

from autotuner import manifest as manifest_mod
from metalbench.bridge import problems, write_job
from metalbench.run import render, fast_p, score


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
    assert loaded.final_benchmark.pairs == 8  # four repeats cannot confirm a win under ~20% on steps this small
    eager = manifest_mod.load(write_job(problem, 2, 2, out=tmp_path / "eager", baseline="plain"))
    assert eager.baseline == "plain"  # the bar the other kernel benchmarks use
    import importlib.util
    spec = importlib.util.spec_from_file_location("job_model", manifest.parent / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    built = module.build()
    inputs = [mx.random.normal(shape) for shape in problem.input_shapes]
    out = built(*inputs)
    mx.eval(out)
    assert isinstance(out, mx.array)


def test_scores_are_the_measured_numbers_and_the_best_kernel_is_reported():
    base = {"choice": "compiled", "clocks_ms": {"p": {"plain": 3.0, "compiled": 2.0}}}
    regions = [{"fingerprint": "big", "ops": ["array.__matmul__"], "p": {"p": 0.8}},
               {"fingerprint": "small", "ops": ["mx.mean", "mx.rsqrt"], "p": {"p": 0.1}, "s": 1.3}]
    hypotheses = [{"region": "big", "verdict": "correct_slower", "region_ms": 2.0, "library_ms": 1.0},
                  {"region": "big", "verdict": "correct_slower", "region_ms": 1.6, "library_ms": 1.0},
                  {"region": "big", "verdict": "failed", "region_ms": 0.1, "library_ms": 1.0},
                  {"region": "small", "verdict": "shipped", "region_ms": 0.5, "library_ms": 1.0}]
    report = {"session": {"status": "complete"}, "baseline": base, "regions": regions, "hypotheses": hypotheses,
              "step_ms": {"p": {"speedup": 1.05, "win_confirmed": False, "speedup_vs_plain": 1.6}}}
    row = score(report, "p")
    # a kernel is installed: the finished model's measured numbers, confirmed or not
    assert row["vs_compiled"] == 1.05 and row["vs_eager"] == 1.6 and row["shipped_regions"] == 1
    # every searched part, costliest first: the win came from the small part, while the big part's kernel
    # lost and was not installed; a failed kernel never counts
    assert row["regions"] == [{"ops": "matmul", "share": 0.8, "speedup": pytest.approx(1 / 1.6), "installed": False},
                              {"ops": "mean+rsqrt", "share": 0.1, "speedup": 2.0, "installed": True}]
    # nothing installed: the finished model is the baseline, whatever noise its final clock read
    nothing = score({"session": {"status": "complete"}, "baseline": base, "regions": regions[:1], "hypotheses": hypotheses[:2],
                     "step_ms": {"p": {"speedup": 1.01, "speedup_vs_plain": 0.95}}}, "p")
    assert nothing["vs_compiled"] == 1.0 and nothing["vs_eager"] == 0.95 and nothing["shipped_regions"] == 0
    # shipped against eager: both columns are still measured
    eager = score({"session": {"status": "complete"}, "baseline": {"choice": "plain", "clocks_ms": {"p": {"plain": 3.0}}},
                   "regions": regions, "hypotheses": hypotheses,
                   "step_ms": {"p": {"speedup": 1.3, "speedup_vs_compiled": 0.9}}}, "p")
    assert eager["vs_eager"] == 1.3 and eager["vs_compiled"] == 0.9
    failed = score({"session": {"status": "failed"}}, "p")
    assert failed["vs_compiled"] is None and failed["vs_eager"] is None and failed["regions"] == []
    rows = [row, nothing, failed]
    assert fast_p(rows, "vs_compiled")["fast_1"] == pytest.approx(1 / 3)  # a failed job counts in the denominator
    assert fast_p(rows, "vs_eager")["fast_1.5"] == pytest.approx(1 / 3)
    text = render({"chip": "test", "problems": {"a": {**row, "set": "standard"}, "b": {**failed, "set": "standard"}}})
    assert "matmul 0.62x (80% of the step); mean+rsqrt 2.00x (10% of the step, installed)" in text and "| n/a | n/a | n/a |" in text


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
