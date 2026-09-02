"""A call on an object that holds model state (a KV cache) is one recorded
state call: a barrier for regions, and replayed by the generated wrapper as
the same call on the same object, so the chains on either side of a cache
write can be delivered. Before this every fusion inside a decoder's
attention block stranded on that write."""

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from autotuner.bind.certify import certify_identity, screen_scope
from autotuner.bind.emit import MODULE_HEADER, Splice, emit_wrapper
from autotuner.bind.verify import verify_retrace
from autotuner.regions.build import build_stretches
from autotuner.regions.roofline import step_floor
from autotuner.measure.peaks import Peaks
from autotuner.trace.replay import replay
from autotuner.trace.types import Retention
from autotuner.trace.walk import state_holders
from autotuner_runtime.kernels import KernelSpec
from autotuner_runtime.swap import flatten_arrays, install, uninstall
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


def traced(model=None, shape=(4, 16)):
    model = model or build_model()
    x = mx.random.normal(shape, key=mx.random.key(5))
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


def layer_scope(trace):
    return next(sc for sc in trace.scope_calls if sc.address == "layer@0")


def certify_layer(model, x, trace):
    layer = layer_scope(trace)
    assert screen_scope(trace, layer.stack) is None
    cls = wrapper_class(emit_wrapper(trace, layer, [], "IdLayer"))

    def install_cb(wrapper):
        occupant = install(model, "layer", wrapper)
        return lambda: uninstall(model, "layer", occupant)

    return certify_identity(build_wrapper=lambda: cls(model.layer, {}),
                            install=install_cb, runs=[lambda: model(x)])


def test_the_cache_method_records_as_one_state_call():
    model, x, trace = traced()
    ops = [n.op for n in trace.nodes]
    assert ops == ["array.__add__", "array.__mul__", "mx.tanh", STATE_OP, "array.sum", "array.__add__"]
    state = trace.nodes[3]
    assert state.in_arrays == trace.nodes[2].out_arrays and len(state.out_arrays) == 1
    receiver = state.scalar_args["receiver"]
    assert (receiver["id"], receiver["path"]) == (id(model.cache), "cache")
    # the ops the method ran, collapsed because one of them wrote the cache
    assert receiver["inner_ops"] == ["array.__getitem__", "array.__setitem__", "array.__getitem__"]
    # the write inside the method is the method's business: nothing the
    # model keeps is a recorded production, and the buffer is a weight
    assert not trace.python_retained() and trace.state_calls() == [3]
    assert "cache.keys" in trace.weight_paths.values()
    assert trace.liveness[state.out_arrays[0]].kind is Retention.CONSUMED
    # the layer scope received the cache as an object argument, after a literal
    layer = layer_scope(trace)
    assert layer.obj_ids == (id(model.cache),)
    assert layer.args_template[1] is None


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


def test_the_scout_line_counts_the_state_a_step_reads_and_the_launches_it_fires():
    """Hiding the cache write must not hide its physics: the bytes the state
    call hands back are state read from memory, and its inner ops are
    launches."""
    _, _, trace = traced()
    f = step_floor(trace, Peaks(bandwidth_gbps=100.0, flops_gflops={"float32": 3000.0}), step_ms=1.0)
    x_bytes, g_bytes, kv_bytes = 4 * 16 * 4, 16 * 4, 5 * 16 * 4
    assert f["bytes_mb"] * 1e6 == x_bytes + 2 * g_bytes + kv_bytes + x_bytes  # + the output
    assert f["launches"] == 3 + 1 + 2  # the chain, the setitem inside the state call, sum and add


def test_the_layer_scope_certifies_with_the_state_call_replayed():
    """The identity wrapper calls cache.update_and_fetch on the object the
    layer was handed, after a literal argument, so the write and the offset
    bump happen for real and three repeated calls match the original bit
    for bit. Its signature must be valid Python: a positional parameter
    after a literal one takes no default."""
    model, x, trace = traced()
    emitted = emit_wrapper(trace, layer_scope(trace), [], "IdLayer")
    assert "def __call__(self, a0, a1, a2):" in emitted.source
    assert "a2.update_and_fetch(" in emitted.source
    result = certify_layer(model, x, trace)
    assert result.ok, result.reason


def test_a_kernel_ships_before_the_cache_write_and_the_state_advances():
    """The chain before the state call becomes one kernel; the retrace shows
    the cut gone, the state call in place, and the cache holding what the
    kernel wrote."""
    model, x, trace = traced()
    before = outputs(model, x)
    chain = trace.nodes[0:3]
    splice = Splice(kernel=FUSED_KERNEL, start_seq=0, end_seq=2,
                    input_ids=(chain[0].in_arrays[0], chain[0].in_arrays[1], chain[1].in_arrays[1]),
                    output_ids=chain[2].out_arrays, fingerprint="pre_cache")
    emitted = emit_wrapper(trace, layer_scope(trace), [splice], "FusedLayer")
    assert "_kernels.try_call(_s, _ins)" in emitted.source
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


class Step(nn.Module):
    """A root that hands its layer a cache object and rewinds it: the shape
    of every decode step, with the cache class chosen per test."""

    def __init__(self, layer, cache):
        super().__init__()
        self.layer = layer
        self.cache = cache

    def __call__(self, x):
        y = self.layer(x, None, self.cache)
        self.cache.offset = 4
        return y


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = mx.ones((16,))

    def __call__(self, x, mask, cache):
        return sum(k.sum(axis=0) for k in flatten_arrays(cache.update_and_fetch(mx.tanh(x * self.g)))) + x


class OwnBuffer:
    """A cache that hands back its buffer itself, as mlx_lm's rotating and
    concatenating caches do."""

    def __init__(self):
        self.keys = mx.zeros((8, 16))
        self.offset = 4

    def update_and_fetch(self, k):
        self.keys[self.offset:self.offset + 1] = k[0:1]
        self.offset += 1
        return self.keys


class Nested(OwnBuffer):
    """A cache that returns a nested structure, as mlx_lm's quantized cache
    does with two triples."""

    def update_and_fetch(self, k):
        keys = super().update_and_fetch(k)
        return (keys[:self.offset], keys[:2]), (keys[:1],)


class Rebinding(OwnBuffer):
    """A cache whose write is a rebinding, not a write in place."""

    def update_and_fetch(self, k):
        self.keys = mx.concatenate([self.keys[:self.offset], k[0:1], self.keys[self.offset + 1:]])
        self.offset += 1
        return self.keys[:self.offset]


class Callable(OwnBuffer):
    """State on an object that is itself callable."""

    def __call__(self, x):
        return x


def test_caches_that_return_their_buffer_or_a_structure_or_rebind_all_deliver():
    for cache_cls in (OwnBuffer, Nested, Rebinding, Callable):
        model = Step(Layer(), cache_cls())
        model, x, trace = traced(model)
        state = [n for n in trace.nodes if n.op.startswith("state:")]
        assert len(state) == 1 and state[0].op == f"state:{cache_cls.__name__}.update_and_fetch", cache_cls
        assert not trace.python_retained(), cache_cls
        result = certify_layer(model, x, trace)
        assert result.ok, (cache_cls, result.reason)
        assert model.cache.offset == 4


class Helper:
    """A stateless helper that holds a table: its method is pure, so its
    ops stay in the record and can be regions, and the step keeps no state."""

    def __init__(self):
        self.table = mx.arange(16, dtype=mx.float32)

    def scale(self, x):
        return x * self.table


class PureLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.helper = Helper()

    def __call__(self, x, mask, cache):
        return mx.tanh(self.helper.scale(x)) + x


def test_a_pure_helper_stays_visible_and_keeps_no_state():
    model = Step(PureLayer(), OwnBuffer())
    model.layer.helper.scale(mx.ones((4, 16)))  # a call before tracing changes nothing below
    model, x, trace = traced(model)
    assert [n.op for n in trace.nodes] == ["array.__mul__", "mx.tanh", "array.__add__"]
    assert not trace.state_calls() and not trace.python_retained()
    assert "layer.helper.table" in trace.weight_paths.values()
    assert dict(state_holders(model))["layer.helper"] is model.layer.helper
    result = certify_layer(model, x, trace)
    assert result.ok, result.reason
