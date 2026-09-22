"""Native targets take the real search, correctness, bind and export paths."""
from dataclasses import replace
from pathlib import Path
import copy
import json

import mlx.core as mx
import pytest

from autotuner.loop import JobRunner, RegionRun, kernel_from_proposal
from autotuner.measure.session import Session
from autotuner.judge.schema import KernelProposal
from autotuner.ladder.static_checks import check
from autotuner_runtime.kernels import LoadedKernel, KernelSpec
from autotuner_runtime.exact import bitwise_equal
from autotuner.e2e import preserving_check


def runner_for(tmp_path):
    model = Path(__file__).parent / 'fixtures/native_recurrence.py'
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(f'''model: {model}
baseline: plain
workloads:
  - name: main
    inputs: [{{shape: [L], dtype: float32}}, {{shape: [L], dtype: float32}}]
primary: {{L: 32}}
sweep: {{L: [16, 32]}}
budget: {{per_region: 2, total: 2}}
''')
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda r: None,
                       clock_pairs=4, session=Session(sleep=lambda s: None))
    runner.load_model()
    runner.trace_workloads()
    regions = runner.build_regions()
    region = next(r for r in regions if r.ops == ('metal_kernel',))
    assert not region.rejected
    runner._capture([region])
    runner.tracer.uninstall()
    region.t_orig_ms['main'] = region.t_rep_ms['main'] = 1.0
    region.p['main'] = 1.0
    return runner, region


def candidate(runner, region):
    seed = runner._build_scaffold(region)
    proposal = KernelProposal(source=seed.source.replace('value * 2', 'value + value'),
                              parent_kernel_id='scaffold', grid=seed.grid,
                              threadgroup=('16', '1', '1'), output_shapes=seed.output_shapes)
    return seed, kernel_from_proposal(runner._contract(region), seed, proposal, 'native_candidate')


def test_native_ladder_and_rejections(tmp_path):
    runner, region = runner_for(tmp_path)
    try:
        seed, spec = candidate(runner, region)
        assert spec.native_call == seed.native_call
        assert check(spec, runner._contract(region)) == []
        roundtrip = KernelSpec.from_json(spec.to_json())
        assert roundtrip == spec
        result = runner._evaluate_kernel(region, spec, 'preserving', run_clock=False)
        assert result.outcome == 'correct_slower', result
        assert result.detail['fallback_engaged']  # smaller sweep uses the original
        wrong = replace(spec, kernel_id='wrong_state', name='wrong_state',
                        source=spec.source.replace('next_state[i] = value', 'next_state[i] = value + 0.000001f'))
        result = runner._evaluate_kernel(region, wrong, 'preserving', run_clock=False)
        assert result.outcome == 'failed', result
        result = runner._evaluate_kernel(region, spec, 'changing', run_clock=False)
        assert result.outcome == 'correct_slower', result
        assert result.detail['correctness_rule'] == 'tolerance'
        changed = copy.deepcopy(spec.native_call)
        changed['bindings'][-1]['scalar'] = 1
        assert check(replace(spec, native_call=changed), runner._contract(region))[0].check == 'native_contract'
        extra = replace(spec, output_names=spec.output_names + ('tmp0',),
                        output_shapes=spec.output_shapes + (('1',),), output_dtypes=spec.output_dtypes+('float32',))
        assert check(extra, runner._contract(region))[0].check == 'native_contract'
    finally:
        runner.tracer.uninstall()


def test_recurrent_trajectory_and_exact_comparison(tmp_path):
    runner, region = runner_for(tmp_path)
    try:
        _, spec = candidate(runner, region)
        kernel = LoadedKernel(spec)
        x, state = runner.tensors['main']
        a = state; b = state
        for _ in range(20):
            original, a = runner.model(x, a)
            proposed, b = kernel([x, b])
            assert bitwise_equal(original, proposed) and bitwise_equal(a, b)
        assert kernel.fallback_fires([mx.ones((16,)), mx.ones((16,))])
        assert not preserving_check(lambda: mx.array([0.0]), lambda: mx.array([-0.0]), 'zero', exact=True).passed
    finally:
        runner.tracer.uninstall()


def test_native_install_final_check_and_fresh_export(tmp_path, monkeypatch):
    from autotuner.ladder.gates import LadderResult
    from autotuner.bind.verify import verify_retrace
    from autotuner_runtime.stats import PairedComparison
    runner, region = runner_for(tmp_path)
    try:
        _, spec = candidate(runner, region)
        # Only the speed verdict is controlled. Real GPU correctness, binding,
        # retrace, model timing, checkpoint and fresh export must still work.
        monkeypatch.setattr(runner, '_model_win', lambda result: all(c.passed for c in result.checks))
        # This equivalent tiny kernel is not a planted speedup. Control all
        # speed decisions, including the final regression veto and sequence
        # confirmation; keep real execution and correctness checks intact.
        monkeypatch.setattr(PairedComparison, 'wins_by', lambda self, margin_ms: True)
        monkeypatch.setattr(PairedComparison, 'loses_by', lambda self, margin_ms: False)
        result = LadderResult('tentative_ship', None, {}, .01, .02, .01, 0., [])
        assert runner._bind_and_promote(RegionRun(region), spec, result)
        assert runner.installed and runner.report.accepted
        runner.tracer.install()
        trace, _ = runner.tracer.trace(runner.model, runner.tensors['main'])
        spans = sorted(runner.cuts['main'])
        checked = verify_retrace(runner.traces['main'], trace, spans,
                                 [runner.cuts['main'][s] for s in spans])
        assert checked.ok, checked.reasons
        runner.tracer.uninstall()
        x, state = runner.tensors['main']
        a = state; b = state
        for _ in range(20):
            y, a = runner.baseline_model(x, a)
            z, b = runner.model(x, b)
            assert bitwise_equal(y, z) and bitwise_equal(a, b)
        # Exercise the fallback on a size outside the candidate signature.
        small = runner.sweep_tensors['main@L=16']
        assert all(bitwise_equal(a,b) for a,b in zip(runner.baseline_model(*small),runner.model(*small)))
        runner._final_check()
        assert runner.final_ok
        out = runner.emit_artifact(tmp_path / 'artifact')
        meta = json.loads((out / 'bundle.json').read_text())
        assert meta['correctness_rule'] == 'exact'
        assert (out / 'validate.py').is_file()
    finally:
        runner.tracer.uninstall()


def test_scripted_judge_uses_native_contract(tmp_path, monkeypatch):
    from autotuner.judge.scripted import ScriptedJudge
    runner, region = runner_for(tmp_path)
    try:
        seed, spec = candidate(runner, region)
        judge = ScriptedJudge([
            {'queue': [{'id': 'edit', 'kind': 'launch', 'assoc_tag': 'preserving',
                        'hypothesis': 'use fewer threads per group'}]},
            {'mutations': [], 'kernel': {'source': spec.source, 'parent_kernel_id': 'scaffold',
                                        'grid': list(spec.grid), 'threadgroup': list(spec.threadgroup),
                                        'output_shapes': [list(s) for s in spec.output_shapes]}}])
        evaluate = runner._evaluate_kernel
        monkeypatch.setattr(runner, '_evaluate_kernel',
                            lambda r, k, tag, run_clock=True: evaluate(r, k, tag, run_clock=False))
        run = runner.open_region(region, judge)
        assert run.scaffold is not None
        runner.hypothesis_cycle(run, judge)
        assert run.hypotheses == 2
        assert any('native_call' in kernel for _, meta, _ in judge.seen for kernel in meta['kernels'].values())
        assert 'bit-identical' in judge.seen[0][1]['body']
        assert 'manifest tolerances' in judge.seen[0][1]['body']
        assert any(a['failed_gate'] is None for a in run.attempts.values())
    finally:
        runner.tracer.uninstall()


def test_native_grouping_and_priority_are_explicit(tmp_path):
    from autotuner.regions.build import build_stretches
    from autotuner.regions.fingerprint import fingerprint
    from autotuner.regions.rank import rank
    from autotuner.regions.types import Region
    from autotuner.ladder.static_checks import buffer_count
    runner, region = runner_for(tmp_path)
    try:
        trace = runner.traces['main']
        span = region.members[0]
        node = trace.nodes[span.start_seq]
        definition = copy.deepcopy(node.kernel_definition)
        definition['kwargs']['compile_options'] = {'math_mode': 'fast'}
        nodes = list(trace.nodes); nodes[node.seq] = replace(node, kernel_definition=definition)
        other = replace(trace, nodes=tuple(nodes))
        assert fingerprint(trace, span) != fingerprint(other, span)
        small = runner.sweep_traces['main@L=16']
        small_span = next(s for s in build_stretches(small,'main') if small.nodes[s.start_seq].kernel_definition)
        assert fingerprint(trace,span) != fingerprint(small,small_span)
        seed = runner._build_scaffold(region)
        assert seed.native_call['template'][-1] == ['Width',32]
        changed = replace(seed,ensure_row_contiguous=False)
        assert check(changed,runner._contract(region))[0].check == 'native_contract'
        assert buffer_count(seed,[1,1]) == 5  # three native arguments, two outputs
        smaller = Region('smaller', ('metal_kernel',),p={'main': .01})
        assert rank([smaller,region])[0] is region
    finally:
        runner.tracer.uninstall()


def test_failed_starter_keeps_original_sequence_available_for_search(tmp_path, monkeypatch):
    runner, region = runner_for(tmp_path)
    try:
        seed = runner._build_scaffold(region)
        bad = replace(seed, source=seed.source.replace('value * 2', 'value * 3'))
        assert bad.source != seed.source
        build = runner._build_scaffold
        monkeypatch.setattr(runner, '_build_scaffold', lambda r:
                            build(r) if r.fingerprint in runner.scaffold_overrides else bad)
        evaluate = runner._evaluate_kernel
        monkeypatch.setattr(runner, '_evaluate_kernel',
                            lambda r, k, tag, run_clock=True: evaluate(r, k, tag, run_clock=False))
        run = runner.open_region(region, None)  # no repair judge is needed for the original
        assert run.scaffold.reference_sequence is not None
        assert run.head is run.scaffold and run.shipped is None
        assert any(row['kind'] == 'scaffold_reference_fallback' for row in runner.log.rows())
    finally:
        runner.tracer.uninstall()


def test_json_search_retrieves_old_parent_and_validates_it_on_gpu(tmp_path, monkeypatch):
    from autotuner.judge.client import JsonJudge

    runner, region = runner_for(tmp_path)
    runner.manifest = replace(runner.manifest, budget_per_region=5, budget_total=5)
    _, spec = candidate(runner, region)
    evaluate = runner._evaluate_kernel
    # Test real correctness workers, without making claims from noisy timing.
    monkeypatch.setattr(runner, '_evaluate_kernel',
                        lambda r, k, tag, run_clock=True: evaluate(r, k, tag, run_clock=False))

    class Judge(JsonJudge):
        proposals = 0
        lookups = 0

        def _ask(self, system, messages):
            meta = json.loads(messages[0]['content'])['region_state']
            assert 'directions' not in meta and 'techniques_reference' in meta
            if self.proposals == 4:
                assert meta['inspiration']['kernel_id'] not in meta['kernels']
                if self.lookups == 0:
                    self.old_id = next(row['kernel'] for row in meta['history'] if row['id'] == 'h2')
                    assert self.old_id not in meta['kernels']
                    self.archive_id = meta['experience_archive']['source_id']
                    query = {'id': self.archive_id, 'find': json.dumps(self.old_id) + ': {'}
                else:
                    read = json.loads(messages[-1]['content'])['source_reads'][0]
                    if self.lookups == 1:
                        query = {'id': self.archive_id, 'start': read['matches'][0]['offset'], 'length': 8000}
                    elif self.lookups == 2:
                        entry = json.JSONDecoder().raw_decode(read['text'].split(':', 1)[1].lstrip())[0]
                        query = {'id': entry['source']['source_id']}
                    else:
                        assert read['text'] == spec.source
                        self.proposals += 1
                        return self.propose(read['text'], self.old_id)
                self.lookups += 1
                return json.dumps({'read_source': [query]})
            self.proposals += 1
            return self.propose(spec.source, 'scaffold')

        def propose(self, source, parent):
            return json.dumps({
                'mutations': [{'op': 'insert', 'item': {
                    'id': f'h{i}', 'kind': f'design{i}', 'assoc_tag': 'preserving',
                    'hypothesis': 'Exercise parent selection with a known correct kernel.'}}
                    for i in range(1, 6)] if self.proposals == 1 else [],
                'kernel': {'source': source, 'parent_kernel_id': parent,
                           'item_id': f'h{self.proposals}', 'grid': list(spec.grid),
                           'threadgroup': list(spec.threadgroup),
                           'output_shapes': [list(s) for s in spec.output_shapes]},
                'lesson': 'The previous candidate passed correctness; no timing claim.'
                          if self.proposals > 1 else None,
            })

    judge = Judge()
    try:
        run = runner.open_region(region, judge)
        runner.hypothesis_cycle(run, judge)
        assert judge.proposals == run.hypotheses == runner.total_hypotheses == 5
        assert judge.lookups == 3  # lookups did not consume candidate attempts
        assert len(run.openers) == 4
        results = {r['hypothesis_id']: r for r in run.attempts.values()}
        assert results['h5']['parent'] == results['h2']['kernel_id']
        assert all(results[f'h{i}']['failed_gate'] is None for i in range(1, 6))
        assert len(runner.lessons) == 4
        assert runner.lessons[-1]['evidence'][0] == results['h4']['kernel_id']
        assert not runner.installed  # correctness and archive access never imply a win
    finally:
        runner.tracer.uninstall()
