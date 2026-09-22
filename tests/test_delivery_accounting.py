"""A kernel is measured against what its scope will actually run.

Graph delivery compiles the scope it installs into, and mx.compile fuses on
its own, so compiling alone can move the step. That effect is clocked and
reported by itself; the kernel's whole-model number is taken against the
scope compiled with no kernel, and the final speedup is split the same way.
"""

from pathlib import Path

import pytest

from autotuner_runtime.stats import comparison_from_samples

from autotuner.loop import JobRunner, RegionRun
from autotuner.measure.session import Session
from autotuner_runtime.graph import GraphWrapper
from autotuner_runtime.swap import resolve_value
from tests.test_install import WIN, elementwise

FIXTURE = Path(__file__).parent / "fixtures" / "planted_win.py"


def _runner(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {FIXTURE}
baseline: plain
workloads:
  - name: main
    inputs: [{{shape: [4, 1024], dtype: float32}}]
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                       clock_pairs=4, session=Session(sleep=lambda seconds: None))
    monkeypatch.setattr(runner, "_model_win", lambda e2e: all(c.passed for c in e2e.checks))
    runner.load_model()
    runner.trace_workloads()
    runner.tracer.uninstall()
    return runner


def test_delivery_is_planned_from_the_record_and_sets_every_library_arm(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    regions = {r.ops: r for r in runner.build_regions() if not r.rejected}
    whole = regions[("array.__mul__", "array.__add__", "mx.maximum", "array.__mul__",
                     "array.__add__", "mx.minimum", "array.__sub__", "array.__mul__")]
    part = regions[("mx.maximum",)]
    assert whole.delivery == {"main": "direct"}   # the kernel call alone, nothing to compile
    assert part.delivery == {"main": "graph"} and not part.delivery_reasons
    # The library arm follows the delivery: a compiled scope competes with
    # compiled ops, a bare kernel call with the plain ops it replaces.
    assert part.library_arm("main", "plain") == "compiled" and whole.library_arm("main", "plain") == "plain"
    assert whole.library_arm("main", "compiled") == "compiled"
    for region in (part, whole):
        region.p["main"] = region.t_orig_ms["main"] = region.t_rep_ms["main"] = 1.0
    runner.store.set_count = lambda *a, **k: 0
    kernel = elementwise("max_probe", "out0[i] = metal::max(in0[i], 0.0f);")
    assert runner._ladder_job(part, kernel, "preserving", True).baseline == "compiled"
    assert runner._ladder_job(whole, kernel, "preserving", True).baseline == "plain"


def test_a_graph_kernel_is_clocked_against_its_compiled_scope(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    regions = runner.build_regions()
    region = next(r for r in regions if r.ops == ("mx.maximum",))
    kernel = elementwise("acct_max", "out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::max(in0[i], 0.0f);")
    try:
        assert runner._bind_and_promote(RegionRun(region), kernel, WIN), runner.log.rows()[-1]
        assert isinstance(resolve_value(runner.model, "chain"), GraphWrapper)
        accepted = runner.report.accepted[-1]
        assert accepted["incumbent_compiled_scopes"] == ["chain"]
        assert accepted["timing_baseline"] == "incumbent_with_compiled_scopes"
        runner._final_check()
        split = runner.report.final["delivery"]
        assert split["compiled_scopes"] == ["chain"]
        assert set(split["timings"]["main"]) == {"plain_vs_compiled_identity", "compiled_identity_vs_patched"}
    finally:
        runner.tracer.uninstall()


def test_a_direct_kernel_is_clocked_against_the_plain_scope(tmp_path, monkeypatch):
    """A kernel that is a module's whole calculation deploys as the bare kernel
    call, so it is measured against the plain module it replaces."""
    from tests.test_sibling_bind import _runner as sibling_runner
    runner, definitions = sibling_runner(tmp_path, monkeypatch)
    region, kernel = definitions["front"]
    assert region.delivery == {"main": "direct"}
    try:
        assert runner._bind_and_promote(RegionRun(region), kernel, WIN), runner.log.rows()[-1]
        assert not isinstance(resolve_value(runner.model, "block.lin_in"), GraphWrapper)
        assert runner.report.accepted[-1]["incumbent_compiled_scopes"] == []
        # a bare kernel call in place of one multiply is a call-site tax on a
        # 20 us step, not a win, and the chained step clock resolves that; this
        # test is about the accounting split, so the final veto is stubbed
        monkeypatch.setattr("autotuner.e2e.step_veto",
                            lambda *args, **kwargs: (comparison_from_samples([1.0] * 4, [0.9] * 4), True))
        runner._final_check()
        assert runner.report.final["delivery"] == {"compiled_scopes": [], "timings": {}}
    finally:
        runner.tracer.uninstall()


def test_the_search_starts_by_settling_delivery_and_clocking_compilation_alone(tmp_path, monkeypatch):
    """Before any kernel, every graph scope's identity is certified so the
    plan the clocks follow is the delivery that ships, and the untouched model
    is measured against itself with those scopes compiled and empty."""
    runner = _runner(tmp_path, monkeypatch)
    regions = runner.build_regions()
    runner._clock_steps()
    runner._settle_delivery(regions)
    assert runner.certified_scopes == {"chain"} and not runner.replay_scopes
    clocks = runner.report.baseline["clocks_ms"]["main"]
    assert clocks["plain"] > 0 and clocks["compiled"] > 0
    assert clocks["compiled_vs_plain"]["verdict"] in {"resolved_improvement", "resolved_regression", "inconclusive"}
    # the manifest asked for the plain baseline: measured, reported, not adopted
    assert runner.report.baseline["choice"] == "plain" and runner.report.baseline["compiled_available"]
    settled = [r for r in runner.log.rows() if r["kind"] == "delivery_settled"]
    assert settled == [{**settled[0], "graph": ["chain"], "replay": [], "baseline": "plain", "baseline_scopes": []}]
    assert not runner.installed and runner.model is not runner.baseline_model


def test_a_scope_whose_compiled_identity_differs_settles_on_replay(tmp_path, monkeypatch):
    """The qwen3.5 fixture's recurrent block is not bitwise the original when
    compiled; settling moves its regions to replay before any clock runs."""
    fixture = Path(__file__).parent / "fixtures" / "qwen_cache_model.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {fixture}
baseline: plain
use_library_inference: false
workloads:
  - name: prompt
    context: 0
    inputs: [{{shape: [1, 4], dtype: int32, high: 64}}]
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                       session=Session(sleep=lambda seconds: None))
    runner.load_model()
    runner.trace_workloads()
    runner.tracer.uninstall()
    regions = runner.build_regions()
    block = next(r for r in regions if r.ops == ("array.__add__",) and not r.rejected)
    assert block.delivery == {"prompt": "graph"}
    runner._settle_delivery(regions)
    assert "model.model.layers.0" in runner.replay_scopes
    assert block.delivery == {"prompt": "replay"} and "output changed" in block.delivery_reasons["prompt"]
    assert block.library_arm("prompt", runner.baseline) == "plain"


def test_a_parent_absorbing_an_installed_child_is_measured_against_the_child(tmp_path, monkeypatch):
    """Composition: the parent's incumbent is the parent compiled with the
    child's cut it absorbs and nothing else, so the new cut alone is measured."""
    from tests.test_sibling_bind import _runner as sibling_runner
    runner, definitions = sibling_runner(tmp_path, monkeypatch)
    try:
        for key in ("front", "fused"):
            assert runner._bind_and_promote(RegionRun(definitions[key][0]), definitions[key][1], WIN), \
                runner.log.rows()[-1]
        first, second = runner.report.accepted[-2:]
        assert first["incumbent_compiled_scopes"] == []   # a bare kernel call against the plain module
        assert second["incumbent_compiled_scopes"] == ["block"]
        assert isinstance(resolve_value(runner.model, "block"), GraphWrapper)
    finally:
        runner.tracer.uninstall()


def test_a_region_at_the_top_level_is_rejected_before_any_budget(tmp_path, monkeypatch):
    """Installation swaps a module in for its parent; ops the model runs in
    its own top-level call have no parent, so such a region is named as
    unsupported at discovery instead of failing every bind."""
    (tmp_path / "model.py").write_text(
        "import mlx.core as mx\nimport mlx.nn as nn\n\n"
        "class Inner(nn.Module):\n"
        "    def __call__(self, x):\n        return mx.maximum(x, 0.0)\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n        self.inner = Inner()\n"
        "    def __call__(self, x):\n        return self.inner(x) * 2.0 + 1.0\n\n"
        "def build():\n    return Model()\n")
    (tmp_path / "manifest.yaml").write_text(
        f"model: {tmp_path / 'model.py'}\nbaseline: plain\n"
        "workloads:\n  - name: main\n    inputs: [{shape: [8, 16], dtype: float32}]\n")
    runner = JobRunner(tmp_path / "manifest.yaml", tmp_path / "work", judge_factory=lambda _: None,
                       session=Session(sleep=lambda _: None))
    runner.load_model()
    runner.trace_workloads()
    runner.tracer.uninstall()
    viable = runner.build_regions()
    assert [r.ops for r in viable] == [("mx.maximum",)]
    stranded = [row for row in runner.report.stranded if "top level" in row["reason"]]
    assert sorted(tuple(row["ops"]) for row in stranded) == sorted([
        ("array.__mul__",), ("array.__add__",), ("array.__mul__", "array.__add__"),
        ("mx.maximum", "array.__mul__"), ("mx.maximum", "array.__mul__", "array.__add__")])


def test_library_inference_takes_the_compiled_baseline_and_kernels_compose_into_it(tmp_path, monkeypatch):
    """The default baseline under mlx-lm generation is the model with its
    outermost compilable scopes compiled and empty: it is installed as the
    starting state, every kernel composes into it, every whole-model number
    is taken against it and nothing else, and the artifact carries it."""
    from autotuner.ladder.gates import LadderResult
    fixture = Path(__file__).parent / "fixtures" / "llama_cache_model.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {fixture}
use_library_inference: true
workloads:
  - name: prompt
    context: 0
    inputs: [{{shape: [1, 6], dtype: int32, high: 64}}]
final_benchmark: {{steps: 1, pairs: 4, warmup_steps: 1}}
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None, clock_pairs=4,
                       session=Session(sleep=lambda seconds: None))
    monkeypatch.setattr(runner, "_model_win", lambda e2e: all(c.passed for c in e2e.checks))
    # The wiring is under test, not this fixture's microsecond clocks: every
    # paired comparison reads as a resolved improvement.
    import autotuner.loop as loop
    from autotuner.measure.clocks import comparison_from_samples
    monkeypatch.setattr(loop, "compare", lambda *a, **k: comparison_from_samples([2.0] * 8, [1.0] * 8))
    try:
        runner.load_model()
        runner.trace_workloads()
        runner.tracer.uninstall()
        regions = runner.build_regions()
        runner._clock_steps()
        runner._settle_delivery(regions)
        baseline = runner.report.baseline
        assert baseline["choice"] == "compiled" and baseline["requested"] == "compiled"
        clocks = baseline["clocks_ms"]["prompt"]
        assert clocks["plain"] > 0 and clocks["compiled"] > 0
        assert runner.step_ms["prompt"] == clocks["compiled"]
        settled = [r for r in runner.log.rows() if r["kind"] == "delivery_settled"][-1]
        scopes = settled["baseline_scopes"]
        assert scopes and set(runner.installed) == set(scopes) and not runner._kernels_installed()
        assert all(isinstance(resolve_value(runner.model, p), GraphWrapper) for p in scopes)
        region = next(r for r in regions if r.ops == ("array.__add__",) and not r.rejected)
        kernel = runner._build_scaffold(region)
        region.p["prompt"] = region.t_orig_ms["prompt"] = region.t_rep_ms["prompt"] = 1.0
        assert runner._bind_and_promote(RegionRun(region), kernel,
                                        LadderResult("tentative_ship", None, {}, .1, 1., .1, 0., [])), \
            runner.log.rows()[-1]
        assert runner._kernels_installed()
        # A replay parent absorbing a baseline scope bypasses it; the rest stay.
        carrying = [p for p in runner.installed if p in scopes
                    and kernel.kernel_id in getattr(resolve_value(runner.model, p), "KERNEL_IDS", ())]
        assert carrying, "the kernel composes into the compiled baseline's own wrapper"
        accepted = runner.report.accepted[-1]
        assert accepted["timing_baseline"] == "incumbent_with_compiled_scopes"
        assert set(accepted["incumbent_compiled_scopes"]) >= set(carrying)
        runner._final_check()
        assert "against_plain" not in runner.report.final and "delivery" not in runner.report.final
        artifact = runner._emit_artifact(tmp_path / "artifact")
        import json
        table = json.loads((artifact / "swap_table.json").read_text())
        assert {row["scope_path"] for row in table} >= set(carrying)
    finally:
        runner.tracer.uninstall()
