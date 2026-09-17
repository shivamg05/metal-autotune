"""Declared shapes must all reach the installed kernel and survive export."""

from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from autotuner.bind.emit import emit_wrapper, emit_wrapper_variants
from autotuner.loop import JobRunner, RegionRun
from autotuner.measure.session import Session
from autotuner_runtime.graph import GraphWrapper
from autotuner_runtime.swap import resolve_value
from tests.test_install import WIN, elementwise


def test_identity_wrapper_resolves_weights_inside_plain_dicts():
    from autotuner.trace import Tracer
    from tests.test_bind import build_wrapper_class
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"scale": mx.ones((16,)), "offsets": [mx.ones((16,))]}
        def __call__(self, x):
            return x * self.config["scale"] + self.config["offsets"][0]
    model = nn.Sequential(Layer())
    x = mx.ones((4, 16))
    tracer = Tracer()
    tracer.install()
    try:
        trace, _ = tracer.trace(model, [x])
    finally:
        tracer.uninstall()
    scope = next(s for s in trace.scope_calls if s.address == "layers.0@0")
    emitted = emit_wrapper(trace, scope, [], "DictWeights")
    wrapped = build_wrapper_class(emitted)(model.layers[0])
    assert mx.array_equal(wrapped(x), model(x)).item()


@pytest.mark.parametrize("dtype,factor", [(mx.float16, 2), (mx.float32, 3)])
def test_single_replay_falls_back_for_unrecorded_dtype_or_scalar(dtype, factor):
    from autotuner.trace import Tracer
    from tests.test_bind import build_wrapper_class

    class Layer(nn.Module):
        def __call__(self, x, factor=2):
            if x.dtype == mx.float32:
                x = x + 1
            return x * factor

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = Layer()

        def __call__(self, x):
            return self.layer(x, factor=2)

    model = Model()
    tracer = Tracer()
    tracer.install()
    try:
        trace, _ = tracer.trace(model, [mx.ones(8)])
    finally:
        tracer.uninstall()
    scope = next(sc for sc in trace.scope_calls if sc.address == "layer@0")
    emitted = emit_wrapper(trace, scope, [], "GuardedReplay")
    wrapped = build_wrapper_class(emitted)(model.layer)
    x = mx.ones(8, dtype=dtype)
    assert mx.array_equal(wrapped(x, factor=factor), model.layer(x, factor=factor)).item()


def test_multi_shape_ship_composes_and_exports(tmp_path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "planted_win.py"
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(f"""model: {fixture}
baseline: plain
workloads:
  - name: small
    inputs: [{{shape: [4, 1024], dtype: float32}}]
  - name: large
    inputs: [{{shape: [7, 1024], dtype: float32}}]
""")
    runner = JobRunner(manifest, tmp_path / "work", lambda region: None,
                       session=Session(sleep=lambda _: None))
    # Timing is tested independently; these known-equivalent kernels exercise
    # real certification, graph verification, composition, correctness, and export.
    monkeypatch.setattr(runner, "_model_win", lambda e2e: all(c.passed for c in e2e.checks))
    try:
        runner.load_model()
        runner.trace_workloads()
        regions = runner.build_regions()
        for op, value, kid in [("mx.maximum", 0, "multi_max"), ("mx.minimum", 8, "multi_min")]:
            region = next(r for r in regions if r.ops == (op,))
            metal_op = "max" if op == "mx.maximum" else "min"
            kernel = elementwise(kid, f"out0[i] = (in0[i] != in0[i]) ? in0[i] : metal::{metal_op}(in0[i], {value}.0f);")
            assert runner._bind_and_promote(RegionRun(region), kernel, WIN), runner.log.rows()[-1]
        # Both cuts live in one scope, so one graph wrapper carries both rules
        # and substitutes each exactly once at every recorded shape.
        wrapper = resolve_value(runner.model, "chain")
        assert isinstance(wrapper, GraphWrapper)
        for label, inputs in runner.tensors.items():
            with wrapper.validate_graph() as calls:
                outputs = runner.model(*inputs)
            hits = Counter(row["kernel_id"] for call in calls for row in call if row["hits"])
            assert hits == {"multi_max": 1, "multi_min": 1}, label
            assert mx.array_equal(outputs, runner.baseline_model(*inputs)).item()
        # The deployed scope is one compiled calculation: a retrace still
        # completes, and neither library op runs there any more.
        runner.tracer.install()
        trace, outputs = runner.tracer.trace(runner.model, runner.tensors["small"])
        assert not {"mx.maximum", "mx.minimum"} & {n.op for n in trace.nodes}
        assert mx.array_equal(outputs, runner.baseline_model(*runner.tensors["small"])).item()
        # An unrecorded shape runs the original module, library ops and all.
        unseen = [mx.ones((9, 1024))]
        with wrapper.validate_graph() as calls:
            trace, outputs = runner.tracer.trace(runner.model, unseen)
        assert calls == [] and {"mx.maximum", "mx.minimum"} <= {n.op for n in trace.nodes}
        assert mx.array_equal(outputs, runner.baseline_model(*unseen)).item()
        runner.tracer.uninstall()
        runner._final_check()
        artifact = runner.emit_artifact(tmp_path / "artifact")
        namespace = {}
        exec((artifact / "patch" / "wrappers.py").read_text(), namespace)
        shipped = [v for v in namespace.values() if isinstance(v, type)
                   and issubclass(v, GraphWrapper) and v is not GraphWrapper]
        assert len(shipped) == 1
        assert len(shipped[0].GRAPH_VARIANTS) == 2  # one guarded variant per recorded shape
        assert shipped[0].KERNEL_IDS == ["multi_max", "multi_min"]
    finally:
        runner.tracer.uninstall()


def test_single_shape_keeps_the_existing_emission(tmp_path):
    from tests.test_install import _runner
    runner = _runner(tmp_path, "[4, 1024]")
    try:
        trace = runner.traces["main"]
        scope = next(sc for sc in trace.scope_calls if sc.address == "chain@0")
        assert emit_wrapper_variants([(trace, scope, [])], "One").source == emit_wrapper(
            trace, scope, [], "One").source
    finally:
        runner.tracer.uninstall()
