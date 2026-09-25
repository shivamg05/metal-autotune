"""Whole-model rejection leaves an original, contract-compatible search parent."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import pytest

from autotuner.loop import JobRunner, PromotionResult, kernel_from_proposal, resolve_parent
from autotuner.ladder.gates import LadderResult
from autotuner.ladder.static_checks import check
from autotuner.judge.schema import KernelProposal
from autotuner.regions.types import Region, Stretch
from autotuner.scaffold.native import native_seed
from autotuner.scaffold.symshape import NoScaffold
from autotuner.trace.recorder import ArrayRef
from autotuner.trace.types import TraceNode
from autotuner_runtime.kernels import KernelSpec


def setup_runner(tmp_path, native):
    definition = {'kwargs': {'name': 'original', 'input_names': ['x'],
                              'output_names': ['y'], 'source': 'y[0] = x[0];'}} if native else None
    arguments = {'args': (), 'kwargs': {'inputs': [ArrayRef(0)], 'grid': (1, 1, 1),
        'threadgroup': (1, 1, 1), 'output_shapes': [(1,)], 'output_dtypes': [mx.float32]}} if native else {
            'args': (ArrayRef(0), 2.0), 'kwargs': {}}
    node = TraceNode(0, 'metal_kernel' if native else 'mx.multiply', (0,), (1,),
                     (((1,), 'float32'),), (((1,), 'float32'),), arguments,
                     'layer@0', 0, ('layer@0',), definition)
    specs = {0: ((1,), 'float32'), 1: ((1,), 'float32')}
    trace = SimpleNamespace(nodes=[node], span_specs=lambda *_: specs)
    span = Stretch('main', 0, 0, (0,), (1,), ('layer@0',))
    region = Region('abcdef0123456789', (node.op,), [span], t_orig_ms={'main': 1.0})
    initial = native_seed(trace, span) if native else KernelSpec(
        kernel_id='initial', name='initial', input_names=('in0',), output_names=('out0',),
        source='out0[0] = in0[0] * 2.0f;', output_shapes=(('1',),), output_dtypes=('float32',))
    runner = JobRunner.__new__(JobRunner)
    runner.manifest = SimpleNamespace(seed=0)
    runner.traces = {'main': trace}
    runner.scaffold_overrides = {}
    runner.kernel_dir = tmp_path / 'kernels'
    runner.log = Mock()
    runner._record_attempt = Mock()
    runner._build_scaffold = lambda r: runner.scaffold_overrides.get(r.fingerprint, initial)
    runner._bind_and_promote = Mock(side_effect=lambda *a, **kw:
        PromotionResult('preflight_passed') if kw.get('preflight') else
        PromotionResult('correctness_failed', 'outputs changed'))
    evaluated = []

    def evaluate(r, kernel, tag, run_clock=True):
        assert check(kernel, runner._contract(r)) == []
        evaluated.append((kernel, tag, run_clock))
        return LadderResult('correct_slower', None, {}, None, None, None, None)

    runner._evaluate_kernel = evaluate
    return runner, region, initial, evaluated


@pytest.mark.parametrize('native', [False, True])
def test_model_rejection_selects_valid_original_with_matching_contract(tmp_path, native):
    runner, region, initial, evaluated = setup_runner(tmp_path, native)
    run = runner.open_region(region, judge=None)
    assert run.close_rule is None
    assert run.head is run.scaffold is runner.scaffold_overrides[region.fingerprint]
    assert run.head.kernel_id.endswith('_original')
    assert evaluated[-1] == (run.head, 'preserving', False)
    assert resolve_parent(run, 'scaffold') is run.head
    assert resolve_parent(run, 'rabcdef0123456789_scaffold').source == initial.source
    assert (runner.kernel_dir / (run.head.kernel_id + '.metal')).exists()
    assert (run.head.native_call is not None) is native
    assert (run.head.reference_sequence is not None) is not native
    proposal = KernelProposal(source=run.head.source, parent_kernel_id='head',
                              grid=run.head.grid, threadgroup=run.head.threadgroup,
                              output_shapes=run.head.output_shapes)
    candidate = kernel_from_proposal(runner._contract(region), run.head, proposal, 'next_attempt')
    assert check(candidate, runner._contract(region)) == []
    assert runner._record_attempt.call_args.kwargs['outcome'] == 'rolled_back'


@pytest.mark.parametrize('failure', ['unsupported', 'validation'])
def test_unavailable_original_closes_region_instead_of_reusing_rejected_head(tmp_path, monkeypatch, failure):
    runner, region, initial, evaluated = setup_runner(tmp_path, native=False)
    if failure == 'unsupported':
        def no_sequence(*_):
            raise NoScaffold('opaque original')
        monkeypatch.setattr('autotuner.scaffold.native.reference_sequence_seed', no_sequence)
    else:
        evaluate = runner._evaluate_kernel
        def fail_original(r, kernel, tag, run_clock=True):
            if kernel.kernel_id.endswith('_original'):
                return LadderResult('failed', 'smoke', {'reason': 'bad original'}, None, None, None, None)
            return evaluate(r, kernel, tag, run_clock)
        runner._evaluate_kernel = fail_original
    run = runner.open_region(region, judge=None)
    if failure == 'unsupported':
        assert 'opaque original' in run.close_rule
        assert run.head is run.scaffold is None
        assert not evaluated
        return
    assert 'original starter' in run.close_rule
    assert run.head is run.scaffold is None
    assert region.fingerprint not in runner.scaffold_overrides
    assert resolve_parent(run, 'rabcdef0123456789_scaffold').source == initial.source
    assert runner._record_attempt.call_args.kwargs['outcome'] == 'rolled_back'


def test_native_model_rejection_fallback_real_worker(tmp_path, monkeypatch):
    from tests.test_native_search import runner_for
    runner, region = runner_for(tmp_path)
    try:
        evaluate = runner._evaluate_kernel
        monkeypatch.setattr(runner, '_evaluate_kernel',
                            lambda r, k, tag, run_clock=True: evaluate(r, k, tag, run_clock=False))
        monkeypatch.setattr(runner, '_bind_and_promote',
                            lambda *_args, **kwargs: PromotionResult('preflight_passed') if kwargs.get('preflight')
                            else PromotionResult('correctness_failed', 'forced model rejection'))
        run = runner.open_region(region, judge=None)
        assert run.close_rule is None
        assert run.head is run.scaffold
        assert run.head.native_call is not None
        assert check(run.head, runner._contract(region)) == []
        proposal = KernelProposal(source=run.head.source, parent_kernel_id='head',
                                  grid=run.head.grid, threadgroup=run.head.threadgroup,
                                  output_shapes=run.head.output_shapes)
        candidate = kernel_from_proposal(runner._contract(region), run.head, proposal, 'after_rollback')
        result = evaluate(region, candidate, 'preserving', run_clock=False)
        assert result.outcome == 'correct_slower', result
    finally:
        runner.tracer.uninstall()
