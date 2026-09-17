"""Mixed native fusions preserve real wiring, state, fallback and source."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner.regions.build import build_stretches
from autotuner.regions.fingerprint import fingerprint
from autotuner.regions.sweep import locate_span
from autotuner.scaffold import build_scaffold
from autotuner.trace import Tracer
from autotuner.trace.types import Liveness, Retention
from autotuner_runtime.exact import bitwise_equal
from autotuner_runtime.kernels import KernelSpec, LoadedKernel


@pytest.fixture
def recorded():
    tracer = Tracer()
    tracer.install()
    try:
        path = Path(__file__).parent / 'fixtures/native_fusion.py'
        module_spec = importlib.util.spec_from_file_location('fixture_native_fusion', path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        model = module.build()
        inputs = [mx.random.normal((32,), key=mx.random.key(i)) for i in range(2)]
        trace, result = tracer.trace(model, inputs)
        tracer.uninstall()
        yield model, inputs, trace, result
    finally:
        tracer.uninstall()


def _full(trace):
    return next(s for s in build_stretches(trace, 'main')
                if s.start_seq == 0 and s.end_seq == len(trace.nodes) - 1)


def test_native_fusion_discovery_and_original_starter(recorded):
    model, inputs, trace, result = recorded
    span = _full(trace)
    assert len(trace.nodes) == 3
    assert trace.nodes[1].kernel_definition is not None
    # State escapes before the last op, so it is the first region output.
    assert span.output_ids == (trace.nodes[1].out_arrays[1], trace.nodes[2].out_arrays[0])
    seed = build_scaffold(trace, span)
    assert seed.reference_sequence and seed.native_call is None
    assert seed.reference_sequence['nodes'][1]['kernel_definition'] == trace.nodes[1].kernel_definition
    restored = KernelSpec.from_json(seed.to_json())
    assert restored == seed
    loaded = LoadedKernel(restored)
    outs = loaded(inputs)
    assert bitwise_equal(outs[0], result[1])
    assert bitwise_equal(outs[1], result[0])
    assert loaded.fallback_fires([mx.ones((16,)), mx.ones((16,))])
    assert locate_span(trace, span, trace, 'main').output_ids == span.output_ids


def test_real_fused_body_preserves_state_over_steps(recorded):
    model, inputs, trace, _ = recorded
    seed = build_scaffold(trace, _full(trace))
    # Match the MLX intermediate rounding explicitly, without assuming fusion
    # permits FMA or dropping the live native state output.
    candidate = replace(seed, kernel_id='fused', name='fused', reference_sequence=None,
                        source='''uint i = thread_position_in_grid.x;
if (i < 32) { float x = in0[i] * 2.0f; float v = x + in1[i];
float y = v * 2.0f; out0[i] = v; out1[i] = y + 1.0f; }''',
                        grid=('32', '1', '1'), threadgroup=('32', '1', '1'))
    fused = LoadedKernel(candidate)
    a, b = inputs[1], inputs[1]
    for _ in range(20):
        expected, a = model(inputs[0], a)
        b, actual = fused([inputs[0], b])
        assert bitwise_equal(expected, actual)
        assert bitwise_equal(a, b)


def test_unknown_source_and_retained_outputs_remain_barriers(recorded):
    _, _, trace, _ = recorded
    opaque = replace(trace.nodes[1], kernel_definition=None)
    missing = replace(trace, nodes=(trace.nodes[0], opaque, trace.nodes[2]))
    assert all(not(s.start_seq <= 1 <= s.end_seq) for s in build_stretches(missing, 'main'))
    kept = dict(trace.liveness)
    aid = trace.nodes[1].out_arrays[1]
    kept[aid] = Liveness(Retention.PYTHON_RETAINED, kept[aid].consumed_by)
    retained = replace(trace, liveness=kept)
    assert all(not(s.start_seq <= 1 <= s.end_seq) for s in build_stretches(retained, 'main'))


def test_mixed_fingerprint_tracks_native_source_and_live_boundary(recorded):
    _, _, trace, _ = recorded
    span = _full(trace)
    definition = dict(trace.nodes[1].kernel_definition)
    definition['kwargs'] = dict(definition['kwargs'], source=definition['kwargs']['source'] + '\n// new source')
    edited = replace(trace, nodes=(trace.nodes[0], replace(trace.nodes[1], kernel_definition=definition), trace.nodes[2]))
    assert fingerprint(trace, span) != fingerprint(edited, span)
    assert fingerprint(trace, span) != fingerprint(trace, replace(span, output_ids=span.output_ids[::-1]))


def test_mixed_candidate_ladder_binding_and_fresh_bundle(tmp_path, monkeypatch):
    """Actual compiler, numeric checks, installation and fresh-process export.

    Timing acceptance alone is controlled: this tests delivery, not speed.
    """
    import json
    from autotuner.loop import JobRunner, RegionRun, kernel_from_proposal, _kernel_view
    from autotuner.measure.session import Session
    from autotuner.judge.schema import KernelProposal
    from autotuner.ladder.gates import LadderResult
    from autotuner.ladder.static_checks import check
    from autotuner.bind.verify import verify_retrace

    manifest = tmp_path / 'manifest.yaml'
    model_path = Path(__file__).parent / 'fixtures/native_fusion.py'
    manifest.write_text(f'''model: {model_path}
baseline: plain
workloads:
  - name: main
    inputs: [{{shape: [L], dtype: float32}}, {{shape: [L], dtype: float32}}]
primary: {{L: 32}}
sweep: {{L: [16, 32]}}
budget: {{per_region: 1, total: 1}}
''')
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda r: None,
                       clock_pairs=4, session=Session(sleep=lambda s: None))
    try:
        runner.load_model()
        runner.trace_workloads()
        region = next(r for r in runner.build_regions()
                      if r.ops == ('mx.multiply', 'metal_kernel', 'mx.add'))
        runner._capture([region])
        runner.tracer.uninstall()
        region.t_orig_ms['main'] = region.t_rep_ms['main'] = 1.0
        region.p['main'] = 1.0
        seed = runner._build_scaffold(region)
        assert _kernel_view(seed)['reference_sequence'] == seed.reference_sequence
        result = runner._evaluate_kernel(region, seed, 'preserving', run_clock=False)
        assert result.outcome == 'correct_slower', result
        fake_win = LadderResult('tentative_ship', None, {}, .01, .02, .01, 0., [])
        assert not runner._bind_and_promote(RegionRun(region), seed, fake_win)
        proposal = KernelProposal(
            source='''uint i = thread_position_in_grid.x;
if (i < 32) { float x = in0[i] * 2.0f; float v = x + in1[i];
float y = v * 2.0f; out0[i] = v; out1[i] = y + 1.0f; }''',
            parent_kernel_id='scaffold', grid=('32', '1', '1'),
            threadgroup=('32', '1', '1'), output_shapes=seed.output_shapes)
        candidate = kernel_from_proposal(runner._contract(region), seed, proposal, 'mixed_fused')
        assert candidate.reference_sequence is None
        assert candidate.input_signature == seed.input_signature
        assert check(candidate, runner._contract(region)) == []
        result = runner._evaluate_kernel(region, candidate, 'preserving', run_clock=False)
        assert result.outcome == 'correct_slower', result
        assert result.detail['fallback_engaged']
        monkeypatch.setattr(runner, '_model_win', lambda result: all(c.passed for c in result.checks))
        assert runner._bind_and_promote(RegionRun(region), candidate, fake_win)
        runner.tracer.install()
        retrace, _ = runner.tracer.trace(runner.model, runner.tensors['main'])
        spans = sorted(runner.cuts['main'])
        checked = verify_retrace(runner.traces['main'], retrace, spans,
                                 [runner.cuts['main'][s] for s in spans])
        assert checked.ok, checked.reasons
        runner.tracer.uninstall()
        x, state = runner.tensors['main']
        a = b = state
        for _ in range(20):
            y, a = runner.baseline_model(x, a)
            z, b = runner.model(x, b)
            assert bitwise_equal(y, z) and bitwise_equal(a, b)
        small = runner.sweep_tensors['main@L=16']
        assert all(bitwise_equal(a, b) for a, b in zip(runner.baseline_model(*small), runner.model(*small)))
        runner._final_check()
        assert runner.final_ok
        out = runner.emit_artifact(tmp_path / 'artifact')
        assert (out / 'validate.py').is_file()
        assert json.loads((out / 'bundle.json').read_text())['correctness_rule'] == 'exact'
    finally:
        runner.tracer.uninstall()


def test_original_sequence_resolver_distinguishes_invert_from_mutation():
    """Resolving a pure operator must not fail because its name starts __i."""
    from autotuner_runtime.original_sequence import _resolve
    assert _resolve('array.__invert__') is mx.array.__invert__
    assert _resolve('array.__getitem__') is mx.array.__getitem__
    with pytest.raises(ValueError, match='mutating operation'):
        _resolve('array.__iadd__')
