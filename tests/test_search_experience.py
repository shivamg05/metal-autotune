"""Compact briefings keep prior successes, failures and exact code recoverable."""
import copy
import json
from types import SimpleNamespace

import pytest

from autotuner.judge.client import JsonJudge
from autotuner.judge.source_context import SourceContext
from autotuner.judge.briefing import share_context
from autotuner.loop import JobRunner, RegionRun
from tests.test_search_feedback import region


def state(count=40, form='single'):
    kernels, history, lessons = {}, [], []
    for i in range(count):
        kid = f'k{i}'
        kernel = {'source': f'out0[0] = in0[0]; // candidate {i}', 'header': '',
                  'hypothesis_id': f'h{i}', 'grid': ['1']*3, 'threadgroup': ['1']*3,
                  'output_shapes': [['1']], 'parent': ('k0' if i <= 4 else f'k{i-1}') if i else None}
        if form == 'native':
            kernel.update(header='// supporting implementation\n'*1800,
                          native_call={'ensure_row_contiguous': True})
        elif form == 'stages':
            kernel['stages'] = [{'inputs': ['in0'], 'outputs': ['out0'],
                                 'source': kernel['source'], 'header': '// stage helpers\n'*900}]
        kernels[kid] = kernel
        history.append({'id': f'h{i}', 'kernel': kid, 'kind': f'design{i%4}',
                        'parent': kernel['parent'], 'hypothesis': 'Test a different layout. '*30,
                        'verdict': 'failed' if i%3 == 0 else 'correct_slower',
                        'summary': 'compile failed' if i%3 == 0 else 'correct but no model win',
                        'detail': {'compile': f'diagnostic for h{i}'}})
        lessons.append({'region': 'r', 'ops': ['mul'], 'lesson': f'Observation {i}. '*100,
                        'evidence': [kid], 'results': {kid: history[-1]}})
    return {'region': {'fingerprint': 'r', 'ops': [{'op': 'mul'}]},
            'scaffold': 'k0', 'head': 'k1', 'shipped': 'k2', 'last_verdict': {'kernel_id': f'k{count-1}'},
            'kernels': kernels, 'history': history, 'lessons': lessons, 'queue': []}


def read_all(context, source_id):
    text, start = '', 0
    while start is not None:
        part = context.read({'read_source': [{'id': source_id, 'start': start}]})[0]
        text += part['text']
        start = part['next_start']
    return text


@pytest.mark.parametrize('form', ['single', 'native', 'stages'])
@pytest.mark.parametrize('count', [10, 75, 200])
def test_history_is_bounded_and_every_old_candidate_remains_readable(form, count):
    meta = state(count, form)
    original = copy.deepcopy(meta)
    context = SourceContext()
    brief = context.focus(meta)
    assert meta == original
    assert len(brief['history']) <= 12
    assert len(brief['lessons']) == 6
    assert all(len(note['lesson']) <= 400 for note in brief['lessons'])
    assert set(brief['kernels']) == {'k0', 'k1', 'k2', f'k{count-1}'}
    assert {row['id'] for row in brief['history']} >= {'h0', 'h1', 'h2', 'h3'}
    last_failed = max(i for i in range(count) if i%3 == 0)
    assert f'h{last_failed}' in {row['id'] for row in brief['history']}
    archive = json.loads(read_all(context, brief['experience_archive']['source_id']))
    assert archive['history'] == meta['history']
    assert archive['lessons']['lesson_1'] == meta['lessons'][0]
    # A losing, non-head kernel remains an exact, editable parent.
    stored = archive['kernels']['k6']
    assert read_all(context, stored['source']['source_id']) == meta['kernels']['k6']['source']
    assert read_all(context, stored['header']['source_id']) == meta['kernels']['k6']['header']
    if form == 'stages':
        assert read_all(context, stored['stages'][0]['header']['source_id']) == meta['kernels']['k6']['stages'][0]['header']
    assert len(json.dumps(share_context(brief))) < 26000


def test_irrelevant_lessons_are_available_without_flooding_the_briefing():
    meta = state()
    meta['lessons'] += [{'region': 'another', 'ops': ['matmul'], 'lesson': 'Matmul lesson.'}]*20
    meta['lessons'].append({'region': 'similar', 'ops': ['mul'], 'lesson': 'Useful negative result.'})
    context = SourceContext()
    brief = context.focus(meta)
    assert all(note['region'] == 'r' for note in brief['lessons'])
    meta['region']['fingerprint'] = 'new_region'
    brief = context.focus(meta)
    assert any(note['lesson'] == 'Useful negative result.' for note in brief['lessons'])
    archive = json.loads(read_all(context, brief['experience_archive']['source_id']))
    assert len(archive['lessons']) == 61


@pytest.mark.parametrize('hidden', [{'rtol': 1e-5}, {'nested': {'atol': 1e-6}}])
def test_archiving_cannot_bypass_metadata_guard(hidden):
    meta = state()
    meta['history'][8]['detail'] = hidden
    with pytest.raises(ValueError):
        JsonJudge().next(meta, {})


def test_agent_fetches_archived_failure_then_its_code_before_proposing():
    meta = state()
    class Judge(JsonJudge):
        calls = 0
        def _ask(self, system, messages):
            self.calls += 1
            payload = json.loads(messages[0]['content'])
            if self.calls == 1:
                assert 'k6' not in payload['region_state']['kernels']
                self.archive_id = payload['region_state']['experience_archive']['source_id']
                return json.dumps({'read_source': [{'id': self.archive_id, 'find': '"k6": {'}]})
            result = json.loads(messages[-1]['content'])['source_reads'][0]
            if self.calls == 2:
                return json.dumps({'read_source': [{'id': self.archive_id, 'start': result['matches'][0]['offset'], 'length': 1000}]})
            if self.calls == 3:
                import re
                source_id = re.search(r'"source_id": "(source_\d+)"', result['text'])[1]
                return json.dumps({'read_source': [{'id': source_id}]})
            assert result['text'] == meta['kernels']['k6']['source']
            return json.dumps({'mutations': [], 'kernel': {
                'parent_kernel_id': 'k6', 'source': result['text'], 'grid': ['1']*3,
                'threadgroup': ['1']*3, 'output_shapes': [['1']]}})
    judge = Judge()
    response = judge.next(meta, {})
    assert response.kernel.parent_kernel_id == 'k6'
    assert judge.calls == 4


def test_lesson_without_evaluated_candidate_is_not_evidence():
    runner = object.__new__(JobRunner)
    runner.lessons = []
    runner._note_lesson(RegionRun(region('r', 0, 2)), SimpleNamespace(lesson='An untested claim'))
    assert runner.lessons == []


def test_refusals_and_repairs_do_not_displace_opening_designs():
    meta = state(75)
    opening = []
    for i in range(1, 5):
        # Transport/plan failures are not evaluated opening designs. A repair
        # is evaluated, but it still must not take another opener's place.
        opening.extend([
            {'id': f'refused{i}', 'kernel': None, 'parent': None,
             'kind': f'refused_kind{i}', 'verdict': 'failed'},
            {'id': f'opening{i}', 'kernel': f'opening_kernel{i}', 'parent': 'k0',
             'kind': f'opening_kind{i}', 'verdict': 'failed'},
            {'id': f'repair{i}', 'kernel': f'repair_kernel{i}', 'parent': f'opening_kernel{i}',
             'kind': f'repair_kind{i}', 'verdict': 'correct_slower'},
        ])
    meta['history'] = opening + meta['history'][5:]
    brief = SourceContext().focus(meta)
    shown = {row['id'] for row in brief['history']}
    assert {f'opening{i}' for i in range(1, 5)} <= shown
    assert len(brief['history']) <= 12


def test_transport_failure_does_not_hide_the_last_evaluated_candidate():
    meta = state()
    latest = meta['last_verdict']['kernel_id']
    payload = {'region_state': meta,
               'verdict': {'outcome': 'failed', 'failed_gate': 'judge_error',
                           'detail': {'reason': 'temporary transport failure'}}}
    brief = SourceContext().focus(payload)
    assert latest in brief['region_state']['kernels']


def test_inspiration_rotates_without_retries_advancing_it():
    meta = state(5)
    meta['head'] = meta['shipped'] = 'k0'
    seen = []
    for i in range(5, 13):
        brief = SourceContext().focus(meta)
        alternative = brief['inspiration']
        seen.append(alternative['kernel_id'])
        assert alternative['kernel_id'] not in brief['kernels']
        retry = copy.deepcopy(meta)
        retry['history'].append({'id': 'refusal', 'verdict': 'plan_refused'})
        assert SourceContext().focus(retry)['inspiration'] == alternative
        meta['kernels'][f'k{i}'] = {'source': 'out0[0]=in0[0];'}
        meta['history'].append({'id': f'h{i}', 'kernel': f'k{i}', 'parent': 'k1', 'kind': 'refine'})
        meta['last_verdict'] = {'kernel_id': f'k{i}'}
    assert set(seen) == {'k1', 'k2', 'k3', 'k4'}


def test_inspiration_waits_for_openers_and_omits_visible_designs():
    assert 'inspiration' not in SourceContext().focus(state(4))
    meta = state(5)
    meta['head'], meta['shipped'] = 'k1', 'k2'
    meta['last_verdict'] = {'kernel_id': 'k3'}
    brief = SourceContext().focus({'region_state': meta, 'verdict': {'kernel_id': 'k4'}})
    assert 'inspiration' not in brief['region_state']


@pytest.mark.parametrize('form', ['single', 'native', 'stages'])
def test_inspiration_preserves_failure_and_shares_source_budget(form):
    meta = state(5, form)
    meta['kernels']['k3']['source'] *= 500
    context = SourceContext()
    brief = context.focus(meta)
    alternative = brief['inspiration']
    # k3 is a failed opening design, still worth learning from.
    assert alternative['kernel_id'] == 'k3'
    assert alternative['verdict'] == 'failed'
    assert alternative['summary'] == 'compile failed'
    shown = sum(len(part['text']) for part in alternative['code_excerpts'])
    assert 0 < shown <= 2000
    shown += sum(len(part['text']) for entry in brief.get('source_catalog', {}).values()
                 for part in entry['excerpts'])
    assert shown <= 12000
    complete = json.loads(read_all(context, alternative['kernel']['source_id']))
    assert read_all(context, complete['source']['source_id']) == meta['kernels']['k3']['source']
