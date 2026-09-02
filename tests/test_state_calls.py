"""A call on an object that holds model state (a KV cache) is one recorded
state call: a barrier for regions, and replayed by the generated wrapper as
the same call on the same object, so the chains on either side of a cache
write can be delivered. The 13:54 Qwen run stranded 67 of 82 candidates on
that write."""

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx

from autotuner.bind.certify import certify_identity, screen_scope
from autotuner.bind.emit import MODULE_HEADER, Splice, emit_wrapper
from autotuner.bind.verify import verify_retrace
from autotuner.regions.build import build_stretches
from autotuner.trace.replay import replay
from autotuner.trace.types import Retention
from autotuner_runtime.kernels import KernelSpec
from autotuner_runtime.swap import install, uninstall
from tests.conftest import current_tracer, tracer_for_module

_module_tracer = tracer_for_module()

STATE_OP = "state:Cache.update_and_fetch"

# (x + g) * b, then tanh: an add feeding a multiply cannot be contracted into
# a fused multiply-add, and precise:: tanh reproduces the library's bits
FUSED = """\
uint i = thread_position_in_grid.x;
uint c = i % (uint)in0_shape[1];
out0[i] = metal::precise::tanh((in0[i] + in1[c]) * in2[c]);
"""
FUSED_KERNEL = KernelSpec(
    kernel_id="k_pre_cache", name="state_test_pre_cache",
    input_names=("in0", "in1", "in2"), output_names=("out0",),
    source=FUSED, grid=("in0.shape[0] * in0.shape[1]", "1", "1"),
    threadgroup=("min(in0.shape[0] * in0.shape[1], 256)", "1", "1"),
    output_shapes=(("in0.shape[0]", "in0.shape[1]"),), output_dtypes=("float32",),
)


def build_model():
    current_tracer()
    path = Path(__file__).parent / "fixtures" / "state_call.py"
    spec = importlib.util.spec_from_file_location("fixture_state_call", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.build()


def traced():
    model = build_model()
    x = mx.random.normal((4, 16), key=mx.random.key(5))
    trace, _ = current_tracer().trace(model, [x])
    return model, x, trace


def wrapper_class(emitted):
    ns = {}
    exec(compile(MODULE_HEADER + emitted.source, "<generated>", "exec"), ns)
    return ns[emitted.class_name]


def outputs(model, x, calls=3):
    outs = [model(x) for _ in range(calls)]
    mx.eval(outs)
    return outs


def test_the_cache_method_records_as_one_state_call():
    model, x, trace = traced()
    ops = [n.op for n in trace.nodes]
    assert ops == ["array.__add__", "array.__mul__", "mx.tanh", STATE_OP, "array.sum", "array.__add__"]
    state = trace.nodes[3]
    assert state.in_arrays == trace.nodes[2].out_arrays and len(state.out_arrays) == 1
    assert state.scalar_args["receiver"] == {"id": id(model.cache), "path": "cache"}
    # the write inside the method is the method's business: nothing the
    # model keeps is a recorded production, and the buffer is a weight
    assert not trace.python_retained()
    assert "cache.keys" in trace.weight_paths.values()
    assert trace.liveness[state.out_arrays[0]].kind is Retention.CONSUMED
    # the layer scope received the cache as an object argument
    layer = next(sc for sc in trace.scope_calls if sc.address == "layer@0")
    assert layer.obj_ids == (id(model.cache),)


def test_a_state_call_is_a_barrier_and_never_replays_in_process():
    _, _, trace = traced()
    spans = {(s.start_seq, s.end_seq) for s in build_stretches(trace, "w")}
    assert (0, 2) in spans and (4, 5) in spans
    assert not any(a <= 3 <= b for a, b in spans)
    state = trace.nodes[3]
    try:
        replay([state], {state.in_arrays[0]: mx.zeros((4, 16))}, state.out_arrays)
    except RuntimeError as e:
        assert "model's own state" in str(e)
    else:
        raise AssertionError("a state call must not replay in process")


def test_the_layer_scope_certifies_with_the_state_call_replayed():
    """The identity wrapper calls cache.update_and_fetch on the object the
    layer was handed, so the write and the offset bump happen for real and
    three repeated calls match the original bit for bit."""
    model, x, trace = traced()
    layer = next(sc for sc in trace.scope_calls if sc.address == "layer@0")
    assert screen_scope(trace, layer.stack) is None
    emitted = emit_wrapper(trace, layer, [], "IdLayer")
    assert "a1.update_and_fetch(" in emitted.source
    cls = wrapper_class(emitted)

    def install_cb(wrapper):
        occupant = install(model, "layer", wrapper)
        return lambda: uninstall(model, "layer", occupant)

    result = certify_identity(build_wrapper=lambda: cls(model.layer, {}),
                              install=install_cb, runs=[lambda: model(x)])
    assert result.ok, result.reason


def test_a_kernel_ships_before_the_cache_write_and_the_state_advances():
    """The chain before the state call becomes one kernel; the retrace shows
    the cut gone, the state call in place, and the cache holding what the
    kernel wrote."""
    model, x, trace = traced()
    before = outputs(model, x)
    layer = next(sc for sc in trace.scope_calls if sc.address == "layer@0")
    chain = trace.nodes[0:3]
    splice = Splice(kernel=FUSED_KERNEL, start_seq=0, end_seq=2,
                    input_ids=(chain[0].in_arrays[0], chain[0].in_arrays[1], chain[1].in_arrays[1]),
                    output_ids=chain[2].out_arrays, fingerprint="pre_cache")
    emitted = emit_wrapper(trace, layer, [splice], "FusedLayer")
    cls = wrapper_class(emitted)
    occupant = install(model, "layer", cls(model.layer, {FUSED_KERNEL.kernel_id: FUSED_KERNEL}))
    try:
        after = outputs(model, x)
        assert all(mx.array_equal(a, b).item() for a, b in zip(after, before))
        retrace, _ = current_tracer().trace(model, [x])
        assert [n.op for n in retrace.nodes] == ["custom_kernel", STATE_OP, "array.sum", "array.__add__"]
        report = verify_retrace(trace, retrace, [(0, 2)], [FUSED_KERNEL.kernel_id])
        assert report.ok, report.reasons
        written = model.cache.keys[4]
        want = mx.tanh((x + model.layer.g) * model.layer.b)[0]
        assert mx.array_equal(written, want).item()
    finally:
        uninstall(model, "layer", occupant)
    assert all(mx.array_equal(a, b).item() for a, b in zip(outputs(model, x), before))
