"""The e2e floor, patched-vs-original, weight sharing, and the step veto."""

import importlib.util
from pathlib import Path

import mlx.core as mx

from autotuner.bind.emit import MODULE_HEADER, Splice, emit_wrapper
from autotuner.e2e import preserving_check, run_e2e, share_weights
from autotuner.measure.session import Session
from autotuner.trace import Tracer
from autotuner_runtime.kernels import KernelSpec
from autotuner_runtime.swap import install

import pytest

# whole jobs and live models: minutes, not seconds
pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"


def load_model():
    spec = importlib.util.spec_from_file_location("fixture_e2e", FIXTURES / "repeated_layers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def test_share_weights_by_identity():
    a, b = load_model(), load_model()
    shared = share_weights(a, b)
    assert shared >= 8  # 4 layers x (g, w)
    x = mx.random.normal((4, 16), key=mx.random.key(0))
    ya, yb = a(x), b(x)
    mx.eval(ya, yb)
    assert mx.array_equal(ya, yb).item()


def test_preserving_check_passes_identical_and_fails_sabotage():
    model = load_model()
    x = mx.random.normal((4, 16), key=mx.random.key(1))
    good = preserving_check(lambda: model(x), lambda: model(x), "same")
    assert good.passed and good.floor_max_abs == 0.0

    bad = preserving_check(lambda: model(x), lambda: model(x) + 0.05, "sabotaged")
    assert not bad.passed
    assert bad.max_abs > bad.allowance


def test_e2e_on_patched_model_sits_on_floor():
    """The real thing: a kernel-spliced model against a fresh baseline, weights
    shared, outputs on the floor, step veto under the paired discipline."""
    tracer = Tracer()
    tracer.install()
    try:
        baseline_model = load_model()
        patched_model = load_model()
        share_weights(baseline_model, patched_model)

        x = mx.random.normal((4, 16), key=mx.random.key(2))
        trace, _ = tracer.trace(patched_model, [x])
        add_node = next(
            n for n in trace.nodes
            if n.op == "array.__add__" and n.module_address == "layers.0@0"
        )
        kernel = KernelSpec(
            kernel_id="k_e2e_add", name="e2e_test_add",
            input_names=("a", "b"), output_names=("out",),
            source="uint i = thread_position_in_grid.x;\nout[i] = a[i] + b[i];",
            grid=("in0.shape[0] * in0.shape[1]", "1", "1"),
            threadgroup=("min(in0.shape[0] * in0.shape[1], 256)", "1", "1"),
            output_shapes=(("in0.shape[0]", "in0.shape[1]"),),
            output_dtypes=("float32",),
        )
        splice = Splice(
            kernel=kernel, start_seq=add_node.seq, end_seq=add_node.seq,
            input_ids=tuple(add_node.in_arrays), output_ids=tuple(add_node.out_arrays),
        )
        scope = next(sc for sc in trace.scope_calls if sc.address == "layers.0@0")
        emitted = emit_wrapper(trace, scope, [splice], "E2eLayer0")
        ns = {}
        exec(compile(MODULE_HEADER + emitted.source, "<generated>", "exec"), ns)
        install(patched_model, "layers.0", ns["E2eLayer0"](
            patched_model.layers[0], {kernel.kernel_id: kernel}
        ))
    finally:
        tracer.uninstall()
        assert tracer.verify_restored() == []

    session = Session()
    result = run_e2e(
        session, baseline_model, patched_model,
        workloads=[("w", [x])], veto_pairs=8,
    )
    assert result.checks[0].passed, result.checks[0]
    assert result.checks[0].max_abs == 0.0  # elementwise add is order-preserving
    # the veto ran on the paired discipline; whether a lone custom add beats
    # the library's own add by the margin is a measurement, not a fixture
    # property, and a quiet machine can resolve it either way
    assert result.veto is not None and result.veto.median_baseline_ms > 0
