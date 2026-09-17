"""The whole-model fp32 reference a reordering kernel is graded against is the
untouched model itself run at fp32. These pin that it computes the same
answer as the recording-replay engine where that engine works, handles every
number format the model may be built in, keeps packed quantized weights
packed, runs a stateful step without touching the live cache, and refuses a
forward that casts back to low precision rather than passing off a rounded
answer as the truth."""

import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from autotuner.ladder.golden import fp32_reference, golden_outputs
from autotuner.regions.price import capture_boundaries
from autotuner.trace import Tracer

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name):
    spec = importlib.util.spec_from_file_location("fx_" + name, FIXTURES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cast(model, dtype):
    from mlx.utils import tree_map
    model.update(tree_map(lambda a: a.astype(dtype) if mx.issubdtype(a.dtype, mx.floating) else a,
                          model.parameters()))
    return model


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_matches_the_replay_engine_in_every_float_format(dtype):
    """On a stateless model the recording replay still works, so it is the
    independent check: the live fp32 run must give the same fp32 answer."""
    model = _cast(_fixture("repeated_layers").build(), dtype)
    x = mx.random.normal((4, 16), key=mx.random.key(3)).astype(dtype)
    tracer = Tracer()
    tracer.install()
    try:
        trace, _ = tracer.trace(model, [x])
        bindings = capture_boundaries(tracer, model, [x], trace, set(trace.inputs) | set(trace.weights))
        replay = golden_outputs(trace.nodes, bindings, trace.step_outputs)[trace.step_outputs[0]]
        live = fp32_reference(model, [x], tracer)[0]
    finally:
        tracer.uninstall()
    assert live.dtype == mx.float32 and replay.dtype == mx.float32
    assert mx.allclose(live, replay, atol=1e-5, rtol=1e-5).item()
    # the model is restored to its own format, and the reference is a
    # genuinely higher-precision answer for the low-precision formats
    assert model.layers[0].w.dtype == dtype
    y = model(x)
    mx.eval(y)
    assert y.dtype == dtype
    if dtype != mx.float32:
        assert not mx.array_equal(y.astype(mx.float32), live).item()
    else:
        assert mx.array_equal(y, live).item()


def test_quantized_weights_stay_packed_and_unpack_in_fp32():
    """A 4-bit model: the packed weights are ints, so they are never lifted
    (this is what keeps the reference small on an 8B model); the matmul
    unpacks on the fly in fp32 and matches dequantize-then-matmul exactly."""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 16, bias=False)

        def __call__(self, x):
            return self.proj(x) + 0.125

    model = Model()
    model.proj.weight = mx.random.normal((16, 64), key=mx.random.key(1)).astype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)
    packed = model.proj.weight
    x = mx.random.normal((2, 64), key=mx.random.key(2)).astype(mx.bfloat16)
    tracer = Tracer()
    tracer.install()
    try:
        golden = fp32_reference(model, [x], tracer)[0]
    finally:
        tracer.uninstall()
    assert packed.dtype == mx.uint32 and model.proj.weight is packed  # never lifted, restored
    dense = mx.dequantize(model.proj.weight, model.proj.scales.astype(mx.float32),
                          model.proj.biases.astype(mx.float32), group_size=64, bits=4)
    expected = x.astype(mx.float32) @ dense.T + 0.125
    assert golden.dtype == mx.float32
    assert mx.allclose(golden, expected, atol=1e-5).item()


def test_a_forward_that_casts_back_to_low_precision_is_refused():
    """A model that hard-casts inside its own forward cannot be run at fp32
    from outside. The reference must say so, not hand back a rounded answer."""
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = mx.random.normal((16, 16), key=mx.random.key(1)).astype(mx.bfloat16)

        def __call__(self, x):
            return (x.astype(mx.bfloat16) @ self.w).astype(mx.bfloat16)

    model = Model()
    x = mx.random.normal((2, 16), key=mx.random.key(2)).astype(mx.bfloat16)
    tracer = Tracer()
    tracer.install()
    try:
        with pytest.raises(ValueError, match="cannot run its math in fp32"):
            fp32_reference(model, [x], tracer)
    finally:
        tracer.uninstall()
    assert model.w.dtype == mx.bfloat16  # restored even on refusal


def test_a_stateful_step_runs_at_fp32_and_leaves_the_live_cache_untouched():
    """The decode case: the model reads and writes a KV cache. The reference
    runs that step against lifted copies of the cache's arrays; the live
    cache keeps its format, position and buffer, and the step stays
    repeatable afterwards. Built twice, the reference is identical."""
    from autotuner_runtime.state import context_step
    fx = _fixture("context_model")
    model = _cast(fx.build(), mx.bfloat16)
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    tok = mx.array([[5]], dtype=mx.int32)
    step = context_step(model, 4, tokens, [tok])
    live = step._cache
    y_before = step(tok)
    mx.eval(y_before)
    # The repeatable call restores isolated array handles. Measure identity
    # across the reference operation itself, after that ordinary baseline call.
    keys_before = [c.keys for c in live]
    tracer = Tracer()
    tracer.install()
    try:
        golden = fp32_reference(step, [tok], tracer)[0]
        again = fp32_reference(step, [tok], tracer)[0]
    finally:
        tracer.uninstall()
    assert golden.dtype == mx.float32 and mx.array_equal(golden, again).item()
    assert [c.offset for c in live] == [4, 4]
    assert all(c.keys is k and k.dtype == mx.bfloat16 for c, k in zip(live, keys_before))
    y_after = step(tok)
    mx.eval(y_after)
    assert mx.array_equal(y_before, y_after).item()
    assert y_before.dtype == mx.bfloat16 and not mx.array_equal(y_before.astype(mx.float32), golden).item()


def test_fp32_cache_and_integer_buffers_are_isolated():
    class State:
        def __init__(self):
            self.value = mx.array([2.0])
            self.count = mx.array([3], dtype=mx.int32)

        def update(self, x):
            self.value[0] = x[0]
            self.count[0] = 9
            return self.value + self.count

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._state = State()
            self._buffers = {"bias": mx.array([0.5], dtype=mx.bfloat16)}

        def __call__(self, x):
            return self._state.update(x) + self._buffers["bias"]

    model = Model()
    value, count, buffers = model._state.value, model._state.count, model._buffers
    tracer = Tracer()
    tracer.install()
    try:
        out = fp32_reference(model, [mx.array([7.0])], tracer)[0]
    finally:
        tracer.uninstall()
    assert out.item() == 16.5
    assert model._state.value is value and value.item() == 2.0
    assert model._state.count is count and count.item() == 3
    assert model._buffers is buffers and buffers["bias"].dtype == mx.bfloat16


def test_low_precision_math_inside_mutating_state_is_refused():
    class State:
        def __init__(self):
            self.value = mx.array([0.0])

        def update(self, x):
            self.value = (x.astype(mx.bfloat16) + 1).astype(mx.float32)
            return self.value

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._state = State()

        def __call__(self, x):
            return self._state.update(x)

    model = Model()
    original = model._state.value
    tracer = Tracer()
    tracer.install()
    try:
        with pytest.raises(ValueError, match="cannot run its math in fp32"):
            fp32_reference(model, [mx.array([1.01])], tracer)
        assert "_append_node" not in vars(tracer.recorder)
    finally:
        tracer.uninstall()
    assert model._state.value is original


def test_unverifiable_compiled_math_is_refused_even_with_fp32_output():
    tracer = Tracer()
    tracer.install()
    try:
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fn = mx.compile(lambda x: x.astype(mx.bfloat16).astype(mx.float32))

            def __call__(self, x):
                return self.fn(x)

        with pytest.raises(ValueError, match="hides its arithmetic"):
            fp32_reference(Model(), [mx.array([1.01])], tracer)
    finally:
        tracer.uninstall()


def test_custom_kernel_math_is_refused_as_an_fp32_reference():
    """A custom kernel's arithmetic is not auditable, like a harness
    kernel's, even when it hands back fp32."""
    tracer = Tracer()
    tracer.install()
    try:
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.copy = mx.fast.metal_kernel(
                    name="fp32_copy", input_names=["a"], output_names=["out"],
                    source="uint i = thread_position_in_grid.x;\nout[i] = a[i];\n")

            def __call__(self, x):
                return self.copy(inputs=[x], grid=(x.size, 1, 1), threadgroup=(1, 1, 1),
                                 output_shapes=[x.shape], output_dtypes=[mx.float32])[0]

        with pytest.raises(ValueError, match="hides its arithmetic"):
            fp32_reference(Model(), [mx.array([1.01])], tracer)
    finally:
        tracer.uninstall()


def test_mlx_lm_compiled_swiglu_is_an_fp32_reference():
    import importlib
    from autotuner.trace import Tracer
    from autotuner.ladder.golden import fp32_reference
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import activations
    class Model(nn.Module):
        def __call__(self, x):
            return activations.swiglu(x, x + 1)
    tracer = Tracer()
    tracer.install()
    try:
        # The real runner installs before importing the model's helpers.
        importlib.reload(activations)
        x = mx.array([-2., 0., 3.], dtype=mx.float16)
        result = fp32_reference(Model(), [x], tracer)[0]
    finally:
        tracer.uninstall()
        importlib.reload(activations)
    assert result.dtype == mx.float32
    xf = x.astype(mx.float32)
    assert mx.allclose(result, nn.silu(xf) * (xf + 1)).item()


@pytest.mark.parametrize('bad_cast', [False, True])
def test_sequence_audit_does_not_retain_intermediate_arrays(bad_cast):
    class Sequence(nn.Module):
        def __call__(self, x):
            for i in range(100):
                x = x + 1
                mx.eval(x)
            return x.astype(mx.float16).astype(mx.float32) if bad_cast else x
    model=Sequence();tracer=Tracer();tracer.install()
    try:
        tracer.patcher.wrap_model(model)  # pre-existing wrappers must be safe too
        x=mx.array([0.],dtype=mx.float16)
        if bad_cast:
            with pytest.raises(ValueError,match='cannot run its math in fp32'):
                fp32_reference(model,[x],tracer,check_graph=False)
        else:
            result=fp32_reference(model,[x],tracer,check_graph=False)
            assert result[0].item() == 100
        assert tracer.recorder.nodes == []
        assert tracer.recorder.scope_calls == []
        assert len(tracer.recorder._by_aid) <= 1
        assert not tracer.recorder.armed
        assert 'module_enter' not in vars(tracer.recorder)
    finally:tracer.uninstall()
