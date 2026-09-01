"""M3: the tracer against the fixture zoo. Each fixture exists to exercise one
mechanism; the done-when list from the plan's M3 section is the test list.

The patch surface is process-global state, so one module-scoped Tracer serves
all tests here, and the final test uninstalls and checks exact restoration.
"""

import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner.trace import Tracer
from autotuner.trace.recorder import OPAQUE_OP
from autotuner.trace.replay import replay
from autotuner.trace.types import Retention, TraceIncomplete

FIXTURES = Path(__file__).parent / "fixtures"

_tracer = None


def tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
        _tracer.install()
    return _tracer


def load_fixture(name: str):
    tracer()  # patches must be up before the model file imports
    spec = importlib.util.spec_from_file_location(f"fixture_{name}", FIXTURES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def trace_of(name: str, shape, dtype=mx.float32):
    model = load_fixture(name)
    x = mx.random.normal(shape, key=mx.random.key(0)).astype(dtype)
    return tracer().trace(model, [x])


def test_operator_soup_records_completely():
    trace, outs = trace_of("operator_soup", (4, 8))
    ops = [n.op for n in trace.nodes]
    # dunders, reflected, in-place, matmul, slicing, slice write, comparison all present
    assert "array.__add__" in ops
    assert "array.__rmul__" in ops
    assert "array.__rsub__" in ops
    assert "array.__iadd__" in ops
    assert "array.__matmul__" in ops
    assert "array.__getitem__" in ops
    assert "array.__setitem__" in ops
    assert "array.__gt__" in ops
    assert "array.__neg__" in ops
    assert "mx.zeros_like" in ops
    # every node inside the module carries a non-empty module address
    assert all(n.module_address for n in trace.nodes)
    # the step output is produced by a recorded call
    assert trace.step_outputs
    assert trace.producer_of(trace.step_outputs[0]) is not None


def test_recording_is_lazy():
    before = mx.get_peak_memory()
    trace, outs = trace_of("repeated_layers", (64, 16))
    assert not trace.in_pass_evaluation
    assert len(trace.nodes) > 0


def test_plain_function_model_has_top_level_address():
    trace, _ = trace_of("plain_function", (4, 16))
    assert [n.op for n in trace.nodes] == ["mx.maximum", "array.__matmul__", "array.__add__"]
    assert all(n.module_address == "@0" for n in trace.nodes)


def test_module_addresses_distinguish_layers():
    trace, _ = trace_of("repeated_layers", (4, 16))
    addresses = {n.module_address for n in trace.nodes}
    assert addresses == {f"layers.{i}@0" for i in range(4)}
    per_layer = [sorted(n.op for n in trace.nodes if n.module_address == f"layers.{i}@0") for i in range(4)]
    assert all(p == per_layer[0] for p in per_layer)


def test_witnessed_compile_records_its_plain_body():
    """A compile applied after the tracer installed records as the plain
    function's own ops, so the scope stays replayable."""
    trace, _ = trace_of("compiled_submodule", (4, 8))
    ops = [n.op for n in trace.nodes]
    assert OPAQUE_OP not in ops
    assert "mx.tanh" in ops


def test_unwitnessed_compile_is_one_opaque_call():
    trace, _ = trace_of("opaque_submodule", (4, 8))
    ops = [n.op for n in trace.nodes]
    assert ops.count(OPAQUE_OP) == 1
    # the ops inside the compiled section (tanh, mul) must NOT be recorded
    assert "mx.tanh" not in ops
    # the opaque node anchors completeness: its output feeds the next op
    opaque = next(n for n in trace.nodes if n.op == OPAQUE_OP)
    assert trace.edges.get(opaque.seq)


def test_cache_retention_marks_python_retained():
    trace, _ = trace_of("cache_retention", (4, 16))
    k_producer = trace.nodes[0]
    assert k_producer.op == "array.__matmul__"
    k_id = k_producer.out_arrays[0]
    assert trace.liveness[k_id].kind is Retention.PYTHON_RETAINED
    assert trace.liveness[k_id].consumed_by  # retained AND consumed, the dangerous case


def test_eager_item_records_completely_with_warning_flag():
    trace, _ = trace_of("eager_item", (4, 8))
    assert trace.in_pass_evaluation
    ops = [n.op for n in trace.nodes]
    assert "array.__matmul__" in ops and "mx.abs" in ops
    assert "array.__truediv__" in ops or "mx.divide" in ops


def test_data_branch_records_only_taken_path():
    trace, _ = trace_of("data_branch", (4, 8))
    ops = [n.op for n in trace.nodes]
    assert trace.in_pass_evaluation  # .item() decided the branch
    assert ("mx.tanh" in ops) != ("mx.sigmoid" in ops)


def test_unwrapped_entry_point_aborts_naming_the_call():
    tr = tracer()

    class Sneaky:
        def __call__(self, x):
            c = mx.array([1.0, 1.0, 1.0, 1.0])  # unpatchable constructor
            return x[:1] + c

    with pytest.raises(TraceIncomplete, match="array.__add__"):
        tr.trace(Sneaky(), [mx.random.normal((4, 4), key=mx.random.key(0))])


def test_step_output_also_intermediate():
    trace, _ = trace_of("step_output_region", (4, 16))
    h_node = next(n for n in trace.nodes if n.op == "mx.maximum")
    h_id = h_node.out_arrays[0]
    assert h_id in trace.step_outputs
    assert trace.liveness[h_id].kind is Retention.STEP_OUTPUT
    assert trace.liveness[h_id].consumed_by


def test_norm_three_projections_share_input():
    trace, _ = trace_of("norm_three_proj", (4, 32))
    norm = next(n for n in trace.nodes if n.op == "mx.fast.rms_norm")
    consumers = trace.edges[norm.seq]
    assert len(consumers) == 3


def test_replay_reproduces_library_outputs_bitwise():
    from autotuner.trace.walk import arrays_by_path

    model = load_fixture("norm_three_proj")
    x = mx.random.normal((4, 32), key=mx.random.key(7))
    trace, outs = tracer().trace(model, [x])
    by_path = arrays_by_path(model)
    bindings = {aid: x for aid in trace.inputs}
    for aid in trace.weights:
        if aid in trace.weight_paths and trace.weight_paths[aid] in by_path:
            bindings[aid] = by_path[trace.weight_paths[aid]]
    replayed = replay(trace.nodes, bindings, trace.step_outputs)
    mx.eval(list(replayed.values()))
    for got, want in zip([replayed[a] for a in trace.step_outputs], list(outs)):
        assert mx.array_equal(got, want).item()


def test_setitem_renames_ssa():
    trace, _ = trace_of("operator_soup", (4, 8))
    setitem = next(n for n in trace.nodes if n.op == "array.__setitem__")
    buf_before = setitem.in_arrays[0]
    buf_after = setitem.out_arrays[0]
    assert buf_before != buf_after
    # consumers after the write read the renamed id, not the old one
    later_consumers = trace.liveness[buf_after].consumed_by
    assert later_consumers
    assert all(s > setitem.seq for s in later_consumers)


def test_uninstall_restores_mx_exactly():
    tr = tracer()
    tr.uninstall()
    assert tr.verify_restored() == []
    x = mx.random.normal((4, 4), key=mx.random.key(0))
    y = x + 1.0
    mx.eval(y)
    global _tracer
    _tracer = None
