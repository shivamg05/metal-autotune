"""Region clocks must execute shared custom work once, regardless of output count."""
from dataclasses import asdict, replace
import re

import mlx.core as mx
import pytest

from autotuner.ladder import child
from autotuner.measure.session import Session
from autotuner.regions import price
from autotuner.regions.build import build_stretches
from autotuner.sandbox.protocol import EvalSetSpec, LadderSpec
from autotuner.trace import Tracer
from autotuner.trace.recorder import ArrayRef as A
from autotuner.trace.replay import compile_replay, replay
from autotuner.trace.serialize import nodes_to_json
from autotuner.trace.types import TraceNode
from autotuner_runtime.exact import bitwise_equal
from autotuner_runtime.kernels import KernelSpec
from tests.fixtures.native_fusion import build


@pytest.fixture
def region():
    tracer = Tracer()
    tracer.install()
    try:
        model = build()
        inputs = [mx.arange(32, dtype=mx.float32), mx.full((32,), 3.0)]
        trace, _ = tracer.trace(model, inputs)
        span = next(s for s in build_stretches(trace, 'main')
                    if s.start_seq == 0 and s.end_seq == len(trace.nodes) - 1)
        bindings = price.capture_boundaries(tracer, model, inputs, trace, set(span.input_ids))
    finally:
        tracer.uninstall()
    return trace, span, bindings


def assert_graph(path, outputs, expected_dispatches):
    # Export before eval discards the graph. Two outputs of one CustomKernel
    # share a single primitive; duplicate replays create separate primitives.
    mx.export_to_dot(str(path), *outputs)
    assert len(re.findall(r'label\s*=\s*"CustomKernel', path.read_text())) == expected_dispatches
    mx.eval(outputs)


@pytest.mark.parametrize('consumer', ['pricing', 'validate', 'score'])
def test_region_clock_executes_shared_custom_kernel_once(region, consumer, monkeypatch, tmp_path):
    trace, span, bindings = region
    reference = replay(trace.nodes, bindings, span.output_ids)
    mx.eval(list(reference.values()))

    class ClockInspected(Exception):
        pass

    def inspect_clock(one_pass, sets, *_):
        outputs = one_pass(sets[0])
        assert_graph(tmp_path / 'clock.dot', outputs, 1)
        assert len(outputs) == len(span.output_ids)
        assert all(bitwise_equal(a, reference[i]) for a, i in zip(outputs, span.output_ids))
        raise ClockInspected

    # Stop at the first actual clock arm: no noisy performance assertion,
    # but real tracing, candidate execution and MLX graph compilation.
    session = Session(sleep=lambda _: None)
    if consumer == 'pricing':
        monkeypatch.setattr(price, 'chained_loop', inspect_clock)
        with pytest.raises(ClockInspected):
            price._looped_replay(session, trace, span, [bindings], {}, 1.0, 'compiled')
    else:
        kernel = KernelSpec(
            kernel_id='replay_regression', name='replay_regression',
            input_names=('in0', 'in1'), output_names=('out0', 'out1'),
            source='''uint i = thread_position_in_grid.x;
float x = in0[i] * 2.0f; float v = x + in1[i];
float y = v * 2.0f; out0[i] = v; out1[i] = y + 1.0f;''',
            grid=('32', '1', '1'), threadgroup=('32', '1', '1'),
            output_shapes=(('32',), ('32',)), output_dtypes=('float32', 'float32'))
        saved = {'inputs': bindings, 'outputs': reference}
        monkeypatch.setattr(child, 'load_set', saved.__getitem__)
        monkeypatch.setattr(child, 'saturate_pool', lambda *_: None)
        monkeypatch.setattr(child, 'chained_loop', inspect_clock)
        spec = LadderSpec(
            kernel=asdict(kernel), assoc_tag='preserving', nodes_json=nodes_to_json(trace.nodes),
            input_ids=span.input_ids, output_ids=span.output_ids,
            eval_sets=(EvalSetSpec('main', ('inputs',), ('outputs',), 1.0, False, None),),
            tolerances={}, kappa=1.25, changing_floor=None, min_win_ms=0.0,
            phase=consumer, baseline='compiled')
        with pytest.raises(ClockInspected):
            child._evaluate_ladder(spec, session)


@pytest.mark.parametrize('outputs', [(5,), (8, 5, 0), (5, 5, 8)])
def test_compiled_boundary_order_and_changing_inputs(outputs):
    nodes = [
        TraceNode(0, 'mx.add', (0, 3), (5,), (), (),
                  {'args': (A(0), A(1)), 'kwargs': {}}, '', 0),
        TraceNode(1, 'mx.multiply', (5,), (8,), (), (),
                  {'args': (A(0), 2), 'kwargs': {}}, '', 0),
    ]
    fixed = {0: mx.full((4,), -100.0), 3: mx.full((4,), 2.0)}
    run = compile_replay(nodes, [0], iter(outputs), fixed)
    # Inputs remain live arguments after compilation, and override a fixed
    # binding of the same id. Single, reordered and repeated outputs survive.
    for value in (1.0, 7.0):
        bindings = {0: mx.full((4,), value)}
        expected = replay(nodes, {**fixed, **bindings}, outputs)
        actual = run(bindings)
        assert len(actual) == len(outputs)
        assert all(bitwise_equal(a, expected[i]) for a, i in zip(actual, outputs))


def test_distinct_custom_dispatches_are_preserved(region, tmp_path):
    trace, span, bindings = region
    # A region can legitimately contain two calls to the same custom kernel.
    # Keep both calls while sharing each call's two outputs.
    second = tuple(replace(n, seq=n.seq + len(trace.nodes),
                           in_arrays=tuple(i + 10 for i in n.in_arrays),
                           out_arrays=tuple(i + 10 for i in n.out_arrays)) for n in trace.nodes)
    nodes = trace.nodes + second
    bindings = {**bindings, **{i + 10: a + 4 for i, a in bindings.items()}}
    outputs = span.output_ids + tuple(i + 10 for i in span.output_ids)
    expected = replay(nodes, bindings, outputs)
    actual = compile_replay(nodes, bindings, outputs)(bindings)
    assert_graph(tmp_path / 'two_calls.dot', actual, 2)
    assert all(bitwise_equal(a, expected[i]) for a, i in zip(actual, outputs))
