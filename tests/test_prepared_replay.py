import mlx.core as mx
import pytest
from autotuner.trace.recorder import ArrayRef as A
from autotuner.trace.types import TraceNode
from autotuner.trace.replay import prepare_replay, replay


def node(op, ins, outs, args, kwargs=None):
    return TraceNode(0, op, ins, outs, (), (), {'args': args, 'kwargs': kwargs or {}}, '', 0)


def check(nodes, binds, outputs):
    expected = replay(nodes, {k: mx.array(v) for k, v in binds.items()}, outputs)
    run = prepare_replay(nodes, binds, outputs)
    actual = run({k: mx.array(v) for k, v in binds.items()})
    assert len(actual) == len(expected)
    assert all(mx.array_equal(a, expected[o]).item() for a, o in zip(actual, outputs))
    return run


def test_chain_containers_and_multiple_outputs():
    nodes = [node('mx.split', (0,), (1, 2), (A(0), 2), {'axis': 0}),
             node('mx.concatenate', (1, 2), (3,), ([A(1), A(0)],), {'axis': 0}),
             node('mx.add', (3,), (4,), (A(0), 2))]
    check(nodes, {0: mx.arange(8)}, [4, 1, 0])


def test_mutation_and_slice():
    nodes = [node('array.__setitem__', (0, 1), (2,), (A(0), slice(1, 3), A(1))),
             node('array.__getitem__', (2,), (3,), (A(0), slice(None, None, -1)))]
    check(nodes, {0: mx.arange(4), 1: mx.array([8, 9])}, [2, 3])


def test_qmm_is_native_and_resolved_once(monkeypatch):
    import autotuner.trace.replay as module
    w, scales, biases = mx.quantize(mx.ones((64, 64)), group_size=64, bits=4)
    binds = dict(enumerate([mx.ones((1, 64)), w, scales, biases]))
    nodes = [node('mx.quantized_matmul', (0, 1, 2, 3), (4,), tuple(A(i) for i in range(4)),
                  {'transpose': True, 'group_size': 64, 'bits': 4})]
    run = check(nodes, binds, [4])
    def forbidden(*a, **k): raise AssertionError('interpreter or compiler used during timing')
    monkeypatch.setattr(module, 'resolve', forbidden)
    monkeypatch.setattr(module, '_materialize', forbidden)
    monkeypatch.setattr(mx, 'compile', forbidden)
    assert mx.array_equal(run(binds)[0], mx.full((1, 64), 64)).item()


@pytest.mark.parametrize('op', ['compiled_fn', 'metal_kernel', 'state:KVCache.update_and_fetch'])
def test_unsupported_calls_fail_before_timing(op):
    with pytest.raises(RuntimeError): prepare_replay([node(op, (0,), (1,), (A(0),))], [0], [1])
