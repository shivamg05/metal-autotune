"""Unsupported declared signatures are normal gate failures, never dead workers."""
from dataclasses import asdict, replace

import mlx.core as mx
import pytest

from autotuner.ladder import child
from autotuner.measure.session import Session
from autotuner.sandbox.protocol import EvalSetSpec, LadderSpec
from autotuner_runtime.kernels import KernelSpec


@pytest.mark.parametrize('specialization', ['input_signature', 'input_signatures', 'native_call'])
def test_worker_rejects_automatic_fallback_before_kernel_call(monkeypatch, specialization):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        spec = KernelSpec(kernel_id='unsupported_' + specialization, name='unsupported_signature',
                          input_names=('in0',), output_names=('out0',), source='out0[0] = in0[0];',
                          output_shapes=(('in0.shape[0]',),), output_dtypes=('float32',))
        signature = [[[1], 'float32']]
        if specialization == 'input_signature':
            spec = replace(spec, input_signature=signature)
        elif specialization == 'input_signatures':
            spec = replace(spec, input_signatures=[signature, [[[3], 'float32']]])
        else:
            spec = replace(spec, native_call={'signature': signature, 'factory': {
                'input_names': ['in0'], 'output_names': ['out0']},
                'bindings': [{'array': 0}], 'template': [], 'init_value': None})
        # A secondary declared workload can differ from a fixed original seed.
        # Exercise the same worker launch guard without submitting GPU work.
        tensors = {'inputs': {0: mx.array([1.0, 2.0])}, 'outputs': {1: mx.array([1.0, 2.0])}}
        monkeypatch.setattr(child, 'load_set', lambda path: tensors[path])
        monkeypatch.setattr(child, 'saturate_pool', lambda *_: None)
        monkeypatch.setattr(child, 'Session', lambda: Session(sleep=lambda _: None))
        actual_call = child.call
        attempted = []
        def call(*args, **kwargs):
            attempted.append(True)
            return actual_call(*args, **kwargs)
        monkeypatch.setattr(child, 'call', call)
        job = LadderSpec(kernel=asdict(spec), assoc_tag='preserving', nodes_json='[]',
                         input_ids=(0,), output_ids=(1,),
                         eval_sets=(EvalSetSpec('other_shape', ('inputs',), ('outputs',), 1.0, False, None),),
                         tolerances={}, kappa=1.25, changing_floor=None, min_win_ms=0.0, phase='validate')
        verdict = child.evaluate_ladder(job)
        assert not verdict.passed and verdict.failed_gate == 'static'
        assert 'fallback_on_workload' in verdict.detail['reason']
        assert attempted == []
    finally:
        mx.set_default_device(previous)
