"""Execute newly covered starters against MLX, including fused boundaries."""
import mlx.core as mx
import pytest

from autotuner.scaffold import build_scaffold, uncovered_op
from autotuner_runtime.numeric import tolerance_for
from tests.conftest import tracer_for_module
from tests.test_scaffold import check, cut, lower_swept, traced

_module_tracer = tracer_for_module()


@pytest.mark.parametrize('op', ['sum', 'mean', 'max', 'min', 'softmax', 'logsumexp'])
@pytest.mark.parametrize('axis', [-1, 0, (0, 2), (1, 2), None, ()])
@pytest.mark.parametrize('dtype', [mx.float32, mx.float16, mx.bfloat16])
def test_axes_and_dtypes(op, axis, dtype):
    class Model:
        def __call__(self, x):
            kwargs = {'axis': axis}
            if op != 'softmax':
                kwargs['keepdims'] = True
            return getattr(mx, op)(x, **kwargs)
    model = Model()
    x, trace = traced(model, (3, 5, 7), 91, dtype)
    span = cut(trace, 0, len(trace.nodes) - 1)
    assert uncovered_op([n.op for n in trace.nodes]) is None
    spec = build_scaffold(trace, span)
    check(spec, model, [x], trace, span, bitwise=op in ('max', 'min'),
          rtol=tolerance_for(dtype)[0], atol=tolerance_for(dtype)[1])


@pytest.mark.parametrize('op', ['mean', 'softmax', 'logsumexp'])
@pytest.mark.parametrize('keepdims', [False, True])
@pytest.mark.parametrize('axis', [0, (0, 2), (1, 2), None])
def test_swept_strided_reduction(op, keepdims, axis):
    class Model:
        def __call__(self, x):
            kwargs = {} if op == 'softmax' else {'keepdims': keepdims}
            return getattr(mx, op)(x.transpose(1, 0, 2)[:, :, 1:], axis=axis, **kwargs)
    model = Model()
    _, trace = traced(model, (4, 3, 9), 12)
    spec, sizes = lower_swept(model, lambda b: (b, 3, 9), (0, len(trace.nodes)-1), batches=(8, 7))
    for x, trace, span in sizes:
        check(spec, model, [x], trace, span, bitwise=False)


@pytest.mark.parametrize('op', ['softmax', 'logsumexp'])
def test_stable_and_nonfinite(op):
    class Model:
        def __call__(self, x):
            return getattr(mx, op)(x, axis=-1)
    model = Model()
    from tests.conftest import current_tracer
    from autotuner_runtime.kernels import call
    x = mx.array([[10000., 9999., -10000.], [-float('inf')]*3,
                  [float('inf'), 1., 2.], [float('nan'), 1., 2.]])
    trace, ref = current_tracer().trace(model, [x])
    spec = build_scaffold(trace, cut(trace, 0, len(trace.nodes)-1))
    got = call(spec, [x])[0]
    assert mx.allclose(got, ref, rtol=1e-5, atol=1e-6, equal_nan=True).item()


@pytest.mark.parametrize('dtype', [mx.float32, mx.float16, mx.bfloat16])
def test_constant_square_and_logsoftmax_fusion(dtype):
    class Model:
        def __call__(self, x):
            h = x ** 2 + mx.array([.25, .5, .75], dtype=x.dtype)
            return h - mx.logsumexp(h, axis=-1, keepdims=True)
    model = Model()
    _, trace = traced(model, (4, 3), 8, dtype)
    spec, sizes = lower_swept(model, lambda b: (b, 3), (0, len(trace.nodes)-1), dtype=dtype, batches=(8, 7))
    for x, trace, span in sizes:
        check(spec, model, [x], trace, span, bitwise=False,
              rtol=tolerance_for(dtype)[0], atol=tolerance_for(dtype)[1])


@pytest.mark.parametrize('grouped', [False, True])
def test_normalization_fusion(grouped):
    class Model:
        def __call__(self, x):
            y = x.reshape(-1, 2, 3, 5, 7) if grouped else x
            axes = (2, 3, 4) if grouped else (2, 3)
            mean = mx.mean(y, axis=axes, keepdims=True)
            var = mx.mean((y-mean)**2, axis=axes, keepdims=True)
            return ((y-mean) * mx.rsqrt(var+1e-5)).reshape(-1, 6, 5, 7)
    model = Model()
    _, trace = traced(model, (4, 6, 5, 7), 9)
    spec, sizes = lower_swept(model, lambda b: (b, 6, 5, 7), (0, len(trace.nodes)-1), batches=(8, 7))
    for x, trace, span in sizes:
        check(spec, model, [x], trace, span, bitwise=False)


@pytest.mark.parametrize('shape', [(), (1,), (2, 0)])
@pytest.mark.parametrize('op', ['sum', 'mean'])
def test_scalar_and_empty_reductions(shape, op):
    from tests.conftest import current_tracer
    from autotuner_runtime.kernels import call
    class Model:
        def __call__(self, x):
            return getattr(mx, op)(x)
    x = mx.zeros(shape)
    trace, ref = current_tracer().trace(Model(), [x])
    spec = build_scaffold(trace, cut(trace, 0, len(trace.nodes)-1))
    got = call(spec, [x])[0]
    assert mx.allclose(got, ref, equal_nan=True).item()


def test_square_declares_changed_arithmetic_and_other_powers_are_refused():
    from autotuner.scaffold import NoScaffold
    class Square:
        def __call__(self, x):
            return x ** 2
    for dtype in (mx.float32, mx.float16, mx.bfloat16):
        model = Square()
        x, trace = traced(model, (7, 33), 13, dtype)
        span = cut(trace, 0, len(trace.nodes)-1)
        spec = build_scaffold(trace, span)
        assert spec.reassociates
        check(spec, model, [x], trace, span, bitwise=False,
              rtol=tolerance_for(dtype)[0], atol=tolerance_for(dtype)[1])
    x, trace = traced(lambda x: x ** 1.5, (7, 33), 13)
    with pytest.raises(NoScaffold, match='power-exponent-not-lowered'):
        build_scaffold(trace, cut(trace, 0, len(trace.nodes)-1))


@pytest.mark.parametrize('precise', [False, True])
def test_softmax_fusion_with_matmul_and_scalar(precise):
    class Model:
        def __call__(self, x):
            scores = (x @ x.T) / mx.sqrt(mx.array(7.0))
            return mx.softmax(scores, axis=-1, precise=precise) @ x
    model = Model()
    _, trace = traced(model, (4, 7), 17)
    spec, sizes = lower_swept(model, lambda b: (b, 7), (0, len(trace.nodes)-1), batches=(8, 7))
    for x, trace, span in sizes:
        check(spec, model, [x], trace, span, bitwise=False)
