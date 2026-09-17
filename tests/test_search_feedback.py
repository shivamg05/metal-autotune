"""The judge sees model rejection and spends each region budget in one search."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from autotuner.loop import JobRunner, RegionRun, PromotionResult, _verdict_payload
from autotuner.ladder.gates import LadderResult
from autotuner.judge.prompts import validate_metadata
from autotuner.judge.scripted import ScriptedJudge
from autotuner.regions.types import Region, Stretch
from autotuner_runtime.kernels import KernelSpec


def region(name, start, end):
    return Region(name, ('metal_kernel',), [Stretch('w', start, end, (0,), (1,), ())])


def test_whole_model_failure_reaches_judge_without_replacing_head():
    runner = object.__new__(JobRunner)
    runner._bind_and_promote = lambda *a, **k: PromotionResult(
        'correctness_failed', 'whole-model outputs changed', [{'reason': 'tolerance'}])
    runner._set_head = lambda *args: (_ for _ in ()).throw(AssertionError('bad kernel became head'))
    run = RegionRun(region('r', 0, 2))
    result = LadderResult('tentative_ship', None, {'ship': True}, 1., 4., 3., .01)
    outcome = runner._apply_verdict(run, SimpleNamespace(assoc_tag='changing'), 'candidate', result)
    assert outcome == 'rolled_back'
    payload = _verdict_payload('h', 'candidate', outcome, result)
    validate_metadata(payload)
    assert payload['detail']['model_check']['status'] == 'correctness_failed'
    assert payload['detail']['model_check']['reason'] == 'whole-model outputs changed'
    assert 'ship' not in payload['detail']
    assert payload['detail']['region_nominated'] is True


def test_correct_but_unresolved_candidate_keeps_timing_explanation():
    runner = object.__new__(JobRunner)
    runner._bind_and_promote = lambda *a, **k: PromotionResult('confirmation_failed', 'win did not repeat')
    heads = []
    runner._set_head = lambda *args: heads.append(args[1])
    run = RegionRun(region('r', 0, 2))
    result = LadderResult('tentative_ship', None, {}, 1., 2., 1., .01)
    assert runner._apply_verdict(run, SimpleNamespace(assoc_tag='changing'), 'candidate', result) == 'correct_slower'
    assert result.detail['model_check']['reason'] == 'win did not repeat'
    assert heads == ['candidate']


@pytest.mark.parametrize('finish_after', [None, 2])
def test_region_search_keeps_history_and_spends_more_than_three_attempts(tmp_path, finish_after):
    runner = object.__new__(JobRunner)
    runner.manifest = SimpleNamespace(budget_per_region=5, budget_total=5)
    runner.total_hypotheses = 0
    runner._finish_requested = lambda: finish_after is not None and runner.total_hypotheses >= finish_after
    runner._meta = lambda run, queue, item: {'attempts': run.hypotheses}
    runner._ask_judge = lambda run, phase, call: (call(), None)
    runner._note_lesson = lambda *args: None
    runner._record_attempt = lambda *args, **kwargs: None
    runner._bind_and_promote = lambda *args, **kwargs: PromotionResult('not_tested')
    runner._evaluate_kernel = lambda *args: LadderResult('correct_slower', None, {}, 1., 2., 1., .01)
    runner.kernel_dir = tmp_path
    seed = KernelSpec('seed', 'seed', ('in0',), ('out0',), 'out0[0]=in0[0];')
    runner._kernel_from_proposal = lambda run, region, proposal, name: replace(seed, kernel_id=name)
    run = RegionRun(region('r', 0, 2), scaffold=seed, head=seed, kernels={'seed': seed})
    script = [{'queue': [{'id': f'h{i}', 'kind': 'retile', 'assoc_tag': 'preserving',
                         'hypothesis': 'try a layout'} for i in range(5)]}]
    script += [{'mutations': [], 'kernel': {'source': seed.source, 'parent_kernel_id': 'head',
                'grid': ['1','1','1'], 'threadgroup': ['1','1','1'], 'output_shapes': [['1']]}} for _ in range(5)]
    judge = ScriptedJudge(script)
    runner.hypothesis_cycle(run, judge)
    assert run.hypotheses == runner.total_hypotheses == (finish_after or 5)
    assert [entry[0] for entry in judge.seen].count('seed') == 1
    assert judge.seen[2][2]['hypothesis_id'] == 'h0'
    assert run.close_rule == ('operator requested final validation' if finish_after
                             else "the region's hypothesis budget is spent")


@pytest.mark.parametrize('total,counts', [(6, [5]), (8, [5, 2]),
                                         (12, [5, 6]), (18, [5, 6, 6])])
def test_job_finishes_each_region_before_opening_the_next(tmp_path, monkeypatch, total, counts):
    from autotuner import loop
    runner = object.__new__(JobRunner)
    targets = [region('a', 0, 4), region('b', 10, 12), region('c', 20, 21)]
    for i, target in enumerate(targets):
        target.p['w'] = .4 - .1 * i
    runner.manifest = SimpleNamespace(workloads=[], budget_total=total, budget_per_region=6, defaulted=[])
    runner.report = SimpleNamespace(manifest_path='manifest', coverage={}, session={}, accepted=[], write=lambda path: None)
    runner.work_dir, runner.gpu_busy_at_start = tmp_path, None
    runner.log = SimpleNamespace(append=lambda *a, **kw: None)
    runner.candidates = SimpleNamespace(append=lambda *a: None)
    runner.session = SimpleNamespace(idled_s=0)
    runner.total_hypotheses, runner.pending_regions, runner.installed = 0, [], {'installed': True}
    runner._phase = runner.load_model = runner.trace_workloads = runner._final_check = lambda *a: None
    runner._finish_requested = lambda: False
    runner.build_regions = lambda: targets
    runner.capture_and_price = lambda regions: regions.copy()
    runner._next_regions = lambda **kw: []
    runner.judge_factory = lambda r: object()
    opened, cycles, closed, refreshes = [], [], [], []
    def open_region(r, judge):
        opened.append(r.fingerprint)
        # A starter repair counts toward the full region budget.
        repair = int(r is targets[0])
        runner.total_hypotheses += repair
        return RegionRun(r, scaffold=object(), hypotheses=repair)
    def cycle(run, judge):
        amount = min(6-run.hypotheses, total-runner.total_hypotheses)
        cycles.append((run.region.fingerprint, amount))
        run.hypotheses += amount
        runner.total_hypotheses += amount
        if run.region is targets[0] and run.shipped is None:
            run.shipped = object()
            runner.report.accepted.append({'kernel': 'first_win'})
        if run.hypotheses == 6 or runner.total_hypotheses == total:
            run.close_rule = 'budget spent'
    def refresh(ranked, shipped):
        # Only remaining regions need repricing after the winner closes.
        assert targets[0] not in ranked
        refreshes.append(True)
        return ranked
    runner.open_region, runner.hypothesis_cycle, runner._refresh_after_ship = open_region, cycle, refresh
    runner._record_region_closed = lambda run: closed.append(run.region.fingerprint)
    monkeypatch.setattr(loop, 'rank', lambda regions: sorted(regions, key=lambda r: r.fingerprint))
    runner._run()
    names = ['a', 'b', 'c'][:len(counts)]
    assert opened == closed == names
    assert cycles == list(zip(names, counts))
    assert runner.total_hypotheses == total
    assert runner.report.coverage['selection'] == {'unsearched': 3 - len(counts)}
    assert len(refreshes) == int(total > 6)  # no repricing after the total budget ends


def test_long_lessons_are_preserved_in_log_and_bounded_only_in_context():
    runner = object.__new__(JobRunner)
    runner.lessons = []
    logged = []
    runner.log = SimpleNamespace(append=lambda event, **row: logged.append(row))
    note = 'Keep this measured observation. ' * 100
    runner._note_lesson(RegionRun(region('r', 0, 2)), SimpleNamespace(lesson=note))
    assert logged[0]['lesson'] == note
    assert len(runner.lessons[0]['lesson']) <= 400
    assert runner.lessons[0]['lesson'].endswith('...')
