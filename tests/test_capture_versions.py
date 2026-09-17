"""Captured boundaries must retain the value at their recorded operation."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from autotuner.trace import Tracer
from autotuner.regions.price import capture_boundaries
from autotuner_runtime.exact import bitwise_equal


@pytest.mark.parametrize('dtype', [mx.float16, mx.float32])
@pytest.mark.parametrize('mutate_input', [False, True])
def test_capture_preserves_each_version_before_later_inplace_updates(dtype, mutate_input):
    class Model(nn.Module):
        def __call__(self, x):
            y = x if mutate_input else x * 2
            y += 3
            y *= 4
            y[0] = 0
            return y

    model = Model()
    original = mx.arange(8).astype(dtype)
    tracer = Tracer()
    tracer.install()
    try:
        trace, _ = tracer.trace(model, [mx.array(original)])
        ids = set(trace.inputs) | {a for node in trace.nodes for a in node.out_arrays}
        captured = capture_boundaries(tracer, model, [mx.array(original)], trace, ids)
    finally:
        tracer.uninstall()
    assert not trace.in_pass_evaluation
    assert bitwise_equal(captured[next(iter(trace.inputs))], original)
    value = mx.array(original)
    for node in trace.nodes:
        assert len(node.out_arrays) == 1, node.op
        if node.op == 'array.__mul__': value = value * 2
        elif node.op == 'array.__iadd__': value = value + 3
        elif node.op == 'array.__imul__': value = value * 4
        elif node.op == 'array.__setitem__':
            value = mx.array(value)
            value[0] = 0
        else: raise AssertionError(node.op)
        assert bitwise_equal(captured[node.out_arrays[0]], value), node.op


def test_inplace_identity_wrapper_preserves_values_and_recorded_operations():
    from autotuner.bind.emit import MODULE_HEADER, emit_wrapper
    from autotuner.bind.verify import verify_retrace

    class Step(nn.Module):
        def __call__(self, x):
            y = x * 2
            y += 3
            y *= 4
            y[0] = 0
            return y

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.step = Step()

        def __call__(self, x):
            return self.step(x)

    model = Model()
    x = mx.arange(8).astype(mx.float16)
    tracer = Tracer()
    tracer.install()
    try:
        trace, expected = tracer.trace(model, [x])
        scope = next(s for s in trace.scope_calls if s.address == 'step@0')
        emitted = emit_wrapper(trace, scope, [], 'IdentityStep')
        namespace = {}
        exec(compile(MODULE_HEADER + emitted.source, '<identity>', 'exec'), namespace)
        model.step = namespace[emitted.class_name](model.step, {})
        retrace, actual = tracer.trace(model, [x])
        assert bitwise_equal(actual, expected)
        report = verify_retrace(trace, retrace, [], [])
        assert report.ok, report.reasons
    finally:
        tracer.uninstall()
