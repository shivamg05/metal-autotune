"""A parent-scope cut that spans two sibling children, installed after a
third child already holds a kernel: the FLUX feed-forward shape. In the
2026-09-06 run every one of its seven region wins failed the retrace, because
the parent's replay left out the child's cut and ran the library there; the
composed wrapper keeps it, on the graph path and on the replay fallback."""

from collections import Counter
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner.bind.verify import verify_retrace
from autotuner.ladder.gates import LadderResult
from autotuner.loop import JobRunner, RegionRun
from autotuner.measure.session import Session
from autotuner_runtime.graph import GraphWrapper
from autotuner_runtime.kernels import KernelSpec
from autotuner_runtime.swap import resolve_value


def _kernel(name, inputs, body):
    return KernelSpec(
        kernel_id=name, name=name, input_names=inputs, output_names=("out",),
        source=f"uint i = thread_position_in_grid.x; {body}",
        grid=("in0.shape[0] * in0.shape[1]", "1", "1"), threadgroup=("32", "1", "1"),
        output_shapes=(("in0.shape[0]", "in0.shape[1]"),), output_dtypes=("float32",),
    )


def _runner(tmp_path, monkeypatch):
    model = Path(__file__).parent / "fixtures" / "sibling_children.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {model}
baseline: plain
workloads:
  - name: main
    inputs: [{{shape: [4, 16], dtype: float32}}]
""")
    # only the speed decision is replaced: identity, correctness, the live
    # dispatch, and the literal retrace all still have to pass
    monkeypatch.setattr(JobRunner, "_model_win", lambda self, result: all(c.passed for c in result.checks))
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                       clock_pairs=4, session=Session(sleep=lambda seconds: None))
    runner.load_model()
    runner.trace_workloads()
    regions = runner.build_regions()
    by_ops = {}
    for r in regions:
        by_ops.setdefault(r.ops, []).append(r)
    front = next(r for r in by_ops[("array.__mul__",)] if r.members[0].scope_stack[-1].startswith("block.lin_in"))
    fused = by_ops[("array.__mul__", "array.__mul__")][0]
    assert fused.members[0].scope_stack[-1] == "block@0"  # starts in act, ends in lin_out
    return runner, {
        "front": (front, _kernel("sib_front", ("x", "w"), "out[i] = x[i] * w[i % 16];")),
        "fused": (fused, _kernel("sib_fused", ("a", "b", "w"), "out[i] = (a[i] * b[i]) * w[i % 8];")),
    }


def _bind(runner, definition):
    region, kernel = definition
    return runner._bind_and_promote(
        RegionRun(region=region), kernel,
        LadderResult("tentative_ship", None, {}, 1.0, 2.0, 1.0, 0.0, []),
    )


@pytest.mark.parametrize("delivery", ["graph", "replay"])
@pytest.mark.parametrize("order", [("front", "fused"), ("fused", "front")])
def test_parent_cut_across_siblings_keeps_the_child_kernel(tmp_path, monkeypatch, order, delivery):
    runner, definitions = _runner(tmp_path, monkeypatch)
    if delivery == "replay":
        runner.replay_scopes.add("block")  # what a scope graph insertion refused gets
    try:
        for key in order:
            assert _bind(runner, definitions[key]), runner.log.rows()[-1]
        assert set(runner.installed) == {"block"}
        x = runner.tensors["main"]
        wrapper = resolve_value(runner.model, "block")
        runner.tracer.install()
        retrace, outputs = runner.tracer.trace(runner.model, x)
        assert mx.array_equal(outputs, runner.baseline_model(*x)).item()
        if delivery == "graph":
            # One compiled wrapper carries both cuts and substitutes each once;
            # the deployed block leaves no library multiply behind.
            assert isinstance(wrapper, GraphWrapper)
            assert "array.__mul__" not in {n.op for n in retrace.nodes}
            with wrapper.validate_graph() as calls:
                outputs = runner.model(*x)
            hits = Counter(row["kernel_id"] for call in calls for row in call if row["hits"])
            assert hits == {"sib_front": 1, "sib_fused": 1}
            assert mx.array_equal(outputs, runner.baseline_model(*x)).item()
        else:
            assert not isinstance(wrapper, GraphWrapper)
            cuts = runner.cuts["main"]
            spans = sorted(cuts)
            checked = verify_retrace(runner.traces["main"], retrace, spans, [cuts[s] for s in spans])
            assert checked.ok, checked.reasons
            assert [n.op for n in retrace.nodes if n.op == "custom_kernel"] == ["custom_kernel"] * 2
        runner.tracer.uninstall()
    finally:
        runner.tracer.uninstall()
        assert runner.tracer.verify_restored() == []
