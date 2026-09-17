from types import SimpleNamespace
import json
import pytest
from autotuner.cli import main, _run_lock
from autotuner.loop import JobRunner
from autotuner.e2e import E2EResult
from autotuner_runtime.stats import comparison_from_samples
from tests.test_run_recovery import bare_runner


def test_finish_command_does_not_take_gpu_lock(tmp_path):
    r = bare_runner(tmp_path)
    (tmp_path / 'run.jsonl').touch()
    with _run_lock():
        assert main(['finish', '--work-dir', str(tmp_path)]) == 0
    assert r._close_rule(None) == 'operator requested final validation'
    r._close_rule(None)
    rows = [json.loads(x) for x in (tmp_path / 'run.jsonl').read_text().splitlines()]
    assert sum(x['kind'] == 'search_finish_requested' for x in rows) == 1
    r.hypothesis_cycle(SimpleNamespace(close_rule=None), None)  # no judge call


def test_finish_requires_existing_run(tmp_path):
    with pytest.raises(SystemExit):main(['finish', '--work-dir', str(tmp_path)])
    assert not (tmp_path / 'finish-search.request').exists()


@pytest.mark.parametrize('sequence_wins', [True, False])
def test_inconclusive_single_step_reaches_sequences(tmp_path, monkeypatch, sequence_wins):
    import autotuner.loop as loop
    r=bare_runner(tmp_path)
    r.installed={'layer': object()};r.shipped_tags={};r.baseline_model=r.model=None
    r.session=SimpleNamespace(wait_ready=lambda:None);r.tensors={'main':[]};r.manifest.workloads=[SimpleNamespace(name='main')]
    r._timed_arms=lambda:{}
    tie=comparison_from_samples([10]*10,[9,11]*5)
    monkeypatch.setattr(loop,'run_e2e',lambda *a,**k:E2EResult(veto=tie,workload_vetos={'main':tie}))
    seq=comparison_from_samples([200]*10, [180]*10 if sequence_wins else [199,201]*5)
    seen=[]
    def sequences(checks):
        seen.append(True)
        return {'main':{'win_confirmed':sequence_wins,'speedup':1/seq.median_ratio}}, E2EResult(veto=seq,workload_vetos={'main':seq})
    r._final_sequences=sequences
    if sequence_wins:r._final_check()
    else:
        with pytest.raises(RuntimeError,match='consecutive-step'):r._final_check()
    assert seen == [True]
    assert r.final_ok == sequence_wins
    assert not r.report.final['paired_win_confirmed']


def test_regression_still_blocks_sequences(tmp_path,monkeypatch):
    import autotuner.loop as loop
    r=bare_runner(tmp_path);r.installed={'layer':object()};r.shipped_tags={}
    r.baseline_model=r.model=None;r.session=SimpleNamespace(wait_ready=lambda:None);r.tensors={};r._timed_arms=lambda:{}
    monkeypatch.setattr(loop,'run_e2e',lambda *a,**k:E2EResult(veto_passed=False))
    r._final_sequences=lambda checks:pytest.fail('regression reached sequences')
    with pytest.raises(RuntimeError,match='regression veto'):r._final_check()


def test_finish_goes_to_final_checks_without_opening_region(tmp_path):
    r=bare_runner(tmp_path)
    r.manifest.budget_per_region=40;r.manifest.budget_total=200;r.manifest.defaulted=[]
    r.gpu_busy_at_start=None;r.pending_regions=[];r.installed={'saved':object()}
    r.session=SimpleNamespace(idled_s=0)
    r.load_model=r.trace_workloads=lambda:None
    r.build_regions=lambda:[]
    r.capture_and_price=lambda regions:[object()]
    r.judge_factory=lambda region:pytest.fail('opened a new region')
    seen=[];r._final_check=lambda:seen.append(True)
    (tmp_path/'finish-search.request').touch()
    r._run()
    assert seen == [True]
    assert r.report.coverage['selection']['unsearched'] == 1
