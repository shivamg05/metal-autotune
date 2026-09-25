"""Unlowered regions remain searchable without relaxing replacement checks."""
from dataclasses import replace

import mlx.core as mx
import pytest

from autotuner.loop import JobRunner, RegionRun, kernel_from_proposal
from autotuner.judge.schema import KernelProposal
from autotuner.ladder.gates import LadderResult
from autotuner.measure.session import Session
from autotuner.scaffold import NoScaffold, build_scaffold, lower_naive
from autotuner.regions.build import build_stretches
from autotuner.trace import Tracer
from autotuner_runtime.exact import bitwise_equal
from autotuner_runtime.kernels import LoadedKernel


@pytest.mark.parametrize('kind', ['conv', 'gather', 'dequantize', 'attention', 'large_fusion'])
def test_original_reference_replays_unlowered_operations(kind):
    # Real operations and data, including the oversized single-group path.
    if kind == 'conv':
        inputs = [mx.ones((1, 8, 2)), mx.ones((3, 3, 2))]
        fn = lambda x, w: mx.conv1d(x, w)
    elif kind == 'gather':
        inputs = [mx.arange(32).reshape(8, 4), mx.array([5, 1, 3])]
        fn = lambda x, idx: x[idx]
    elif kind == 'dequantize':
        inputs = list(mx.quantize(mx.ones((8, 64)), bits=4, group_size=64))
        fn = lambda w, s, b: mx.dequantize(w, s, b, bits=4, group_size=64)
    elif kind == 'attention':
        inputs = [mx.ones((1, 2, 4, 16)) for _ in range(3)]
        fn = lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.25)
    else:
        inputs = [mx.ones((1, 1024)), mx.ones((1024, 2048))]
        fn = lambda x, w: mx.matmul(mx.fast.rms_norm(x, None, 1e-5), w)
    mx.eval(inputs)
    tracer = Tracer()
    tracer.install()
    try:
        trace, expected = tracer.trace(fn, inputs)
    finally:
        tracer.uninstall()
    span = next(s for s in build_stretches(trace, 'main')
                if s.start_seq == 0 and s.end_seq == len(trace.nodes) - 1)
    with pytest.raises(NoScaffold):
        lower_naive(trace, span)
    seed = build_scaffold(trace, span)
    assert seed.reference_sequence is not None
    assert bitwise_equal(LoadedKernel(seed)(inputs)[0], expected)
    if kind == 'conv':
        bad = replace(trace, nodes=(replace(trace.nodes[0], op='opaque_unknown'),))
        with pytest.raises(NoScaffold):
            build_scaffold(bad, span)


@pytest.mark.parametrize('split', [False, True])
def test_from_scratch_conv_replacement_checks_binding_and_export(tmp_path, monkeypatch, split):
    # Timing acceptance is controlled. This proves delivery/correctness, not a win.
    model = tmp_path / 'model.py'
    model.write_text('''import mlx.core as mx
import mlx.nn as nn
class Block(nn.Module):
    def __call__(self, x, w):
        y = mx.conv1d(x, w)
        return y, mx.add(y, 1.0)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
    def __call__(self, x, w):
        return self.block(x, w)
def build():
    return Model()
''')
    if split:
        model.write_text(model.read_text().replace('return self.block(x, w)',
            'return self.block(x[:, :-1], w), self.block(x[:, -1:], w)'))
    manifest = tmp_path / 'job.yaml'
    manifest.write_text('''model: model.py
baseline: plain
use_library_inference: false
workloads:
  - name: main
    inputs: [{shape: [1, L, 1], dtype: float32}, {shape: [1, 1, 1], dtype: float32}]
primary: {L: 32}
sweep: {L: [16, 32]}
budget: {per_region: 1, total: 1}
''')
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda _: None,
                       clock_pairs=4, session=Session(sleep=lambda _: None))
    try:
        runner.load_model()
        runner.trace_workloads()
        region = next(r for r in runner.build_regions() if r.ops == ('mx.conv1d', 'mx.add'))
        runner._capture([region])
        runner.tracer.uninstall()
        region.t_orig_ms['main'] = region.t_rep_ms['main'] = 1.0
        region.p['main'] = 1.0
        run = runner.open_region(region, judge=None)
        assert run.close_rule is None
        seed = run.scaffold
        assert seed.reference_sequence is not None
        fake_win = LadderResult('tentative_ship', None, {}, .01, .02, .01, 0., [])
        assert not runner._bind_and_promote(run, seed, fake_win)
        source = '''uint i = thread_position_in_grid.x;
if (i < in0_shape[1]) { float y = in0[i] * in1[0]; out0[i] = y; out1[i] = y + 1.0f; }'''
        proposal = KernelProposal(source=source, parent_kernel_id='scaffold',
            grid=('in0.shape[1]', '1', '1'), threadgroup=('32', '1', '1'),
            output_shapes=(('1', 'in0.shape[1]', '1'),) * 2)
        candidate = kernel_from_proposal(runner._contract(region), seed, proposal, 'conv_fused')
        assert candidate.reference_sequence is None
        if split:
            from autotuner.ladder.static_checks import check
            assert len(candidate.input_signatures) == 2
            assert check(replace(candidate, input_signatures=None), runner._contract(region))[0].check == 'input_signatures'
        result = runner._evaluate_kernel(region, candidate, 'preserving', run_clock=False)
        assert result.outcome == 'correct_slower', result
        assert result.detail['fallback_engaged']  # the unoptimized sweep shape uses the original
        bad = replace(candidate, kernel_id='bad_conv', name='bad_conv',
                      source=source.replace('y + 1.0f', 'y + (in0_shape[1] == 1 ? 2.0f : 1.0f)' if split else 'y + 2.0f'))
        failure = runner._evaluate_kernel(region, bad, 'preserving', run_clock=False)
        assert failure.outcome == 'failed', failure
        assert failure.failed_gate not in ('static', 'compile'), failure
        if split:
            assert failure.failed_gate == 'workloads', failure  # only the one-token copy is wrong
        monkeypatch.setattr(runner, '_model_win', lambda result: all(c.passed for c in result.checks))
        assert runner._bind_and_promote(RegionRun(region), candidate, fake_win)
        runner._final_check()
        assert runner.final_ok
        out = runner.emit_artifact(tmp_path / 'artifact')
        assert (out / 'validate.py').exists()
    finally:
        runner.tracer.uninstall()


def test_reference_validation_failure_never_reaches_judge(tmp_path):
    from unittest.mock import Mock
    from tests.test_scaffold_model_fallback import setup_runner
    from autotuner.scaffold.native import reference_sequence_seed

    runner, region, _, _ = setup_runner(tmp_path, native=False)
    seed = reference_sequence_seed(runner.traces['main'], region.members[0])
    runner._build_scaffold = lambda _: seed
    runner._evaluate_kernel = Mock(return_value=LadderResult(
        'failed', 'smoke', {}, None, None, None, None, []))
    runner._judge_fix = Mock()
    run = runner.open_region(region, judge=None)
    assert run.close_rule == 'original reference failed validation at smoke'
    assert run.scaffold is None
    runner._judge_fix.assert_not_called()


@pytest.mark.parametrize('gate', ['static', 'compile', 'workloads'])
def test_failed_generated_starter_uses_original_before_asking_judge(tmp_path, gate):
    from unittest.mock import Mock
    from autotuner.loop import PromotionResult
    from tests.test_scaffold_model_fallback import setup_runner

    runner, region, _, _ = setup_runner(tmp_path, native=False)
    evaluate = runner._evaluate_kernel
    def fail_generated(r, kernel, tag, run_clock=True):
        if kernel.reference_sequence is None:
            return LadderResult('failed', gate, {}, None, None, None, None, [])
        return evaluate(r, kernel, tag, run_clock)
    runner._evaluate_kernel = fail_generated
    runner._judge_fix = Mock()
    runner._bind_and_promote = Mock(side_effect=lambda *a, **kw:
        PromotionResult('preflight_passed' if kw.get('preflight') else 'not_tested'))
    run = runner.open_region(region, judge=None)
    assert run.close_rule is None
    assert run.scaffold.reference_sequence is not None
    runner._judge_fix.assert_not_called()


@pytest.mark.parametrize('op', ['reshape', 'qmm', 'qmm_norm'])
def test_reference_variants_keep_each_shapes_original_wiring(op):
    # CPU-only: validate dispatch, guards and serialization without competing
    # with an optimization job for GPU time.
    from autotuner.scaffold.native import reference_sequence_seed
    from autotuner.regions.types import Stretch
    from autotuner_runtime.kernels import KernelSpec

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        cases = []
        weights = mx.quantize(mx.ones((16, 64)), group_size=64, bits=4)
        for length in (7, 1):
            x = mx.arange(length * 64, dtype=mx.float32).reshape(1, length, 64) / 128
            if op == 'reshape':
                inputs = [x]
                fn = lambda x: mx.reshape(x, (x.shape[1], 64))
            else:
                inputs = [x, *weights]
                def fn(x, w, s, b):
                    y = mx.quantized_matmul(x, w, s, b, group_size=64, bits=4)
                    return mx.fast.rms_norm(y + 1, None, 1e-5) if op == 'qmm_norm' else y
            tracer = Tracer()
            tracer.install()
            try:
                trace, expected = tracer.trace(fn, inputs)
            finally:
                tracer.uninstall()
            nodes = trace.nodes
            # Include the whole straight-line computation, exposing its last output.
            produced = {a for n in nodes for a in n.out_arrays}
            ids = tuple(dict.fromkeys(a for n in nodes for a in n.in_arrays if a not in produced))
            span = Stretch('main', 0, len(nodes)-1, ids, nodes[-1].out_arrays, ('block@0',))
            cases.append((trace, span, inputs, expected))
        seed = reference_sequence_seed(cases[0][0], cases[0][1],
                                       [(t, s) for t, s, _, _ in cases])
        assert seed.input_signature is None and len(seed.input_signatures) == 2
        loaded = LoadedKernel(KernelSpec.from_json(seed.to_json()))
        for _, _, inputs, expected in cases:
            assert not loaded.fallback_fires(inputs)
            assert bitwise_equal(loaded(inputs)[0], expected)
        unseen = [mx.ones((1, 3, 64)), *cases[0][2][1:]]
        assert loaded.fallback_fires(unseen)
    finally:
        mx.set_default_device(previous)


def test_failed_binding_preflight_spends_no_judge_attempts(tmp_path):
    from unittest.mock import Mock
    from autotuner.loop import PromotionResult
    from tests.test_scaffold_model_fallback import setup_runner

    runner, region, _, _ = setup_runner(tmp_path, native=False)
    runner._bind_and_promote = Mock(return_value=PromotionResult('binding_failed', 'missing branch'))
    runner._evaluate_kernel = Mock()
    runner._judge_fix = Mock()
    run = runner.open_region(region, judge=None)
    assert run.close_rule == 'installation preflight failed: missing branch'
    runner._evaluate_kernel.assert_not_called()
    runner._judge_fix.assert_not_called()
    assert runner._bind_and_promote.call_args.kwargs['preflight'] is True
