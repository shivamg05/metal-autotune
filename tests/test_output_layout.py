"""Output defaults stay out of source, and reruns preserve earlier evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_cli_default_is_fresh_and_explicit_paths_still_work(tmp_path, monkeypatch):
    from autotuner import cli
    monkeypatch.chdir(tmp_path)
    paths = []
    monkeypatch.setattr(cli, '_execute', lambda args, parser: paths.append(Path(args.work_dir)) or 0)
    assert cli.main(['run', 'unused.yaml']) == 0
    assert cli.main(['run', 'unused.yaml']) == 0
    assert paths[0].parent == paths[1].parent == Path('runs')
    assert paths[0] != paths[1]
    assert not Path('runs').exists()  # parsing/help must not create run directories
    explicit = tmp_path / 'elsewhere'
    assert cli.main(['run', 'unused.yaml', '--work-dir', str(explicit)]) == 0
    assert paths[-1] == explicit


@pytest.mark.parametrize('exit_code', [0, 1])
def test_benchmark_reruns_preserve_jobs_and_report_output_paths(tmp_path, monkeypatch, exit_code):
    from metalbench import run
    from metalbench.bridge import problems
    monkeypatch.setattr(run, 'chip_name', lambda: 'test-chip')
    monkeypatch.setattr(run, 'problems', lambda sets: {'abs': problems()['abs']})
    works = []

    def execute(command, **kwargs):
        work = Path(command[command.index('--work-dir') + 1])
        assert not work.exists()
        works.append(work)
        work.mkdir(parents=True)
        if not exit_code:
            (work / 'report.json').write_text(json.dumps({'session': {'status': 'complete'},
                'baseline': {'choice': 'compiled'}, 'step_ms': {'abs': {'speedup_vs_plain': 1.0}}}))
        return SimpleNamespace(returncode=exit_code, stdout='progress', stderr='failed' if exit_code else '')

    monkeypatch.setattr(run.subprocess, 'run', execute)
    args = ['--only', 'abs', '--baseline', 'compiled', '--output-dir', str(tmp_path / 'output')]
    assert run.main(args) == 0
    assert run.main(args) == 0  # already scored, no new job
    assert len(works) == 1
    assert run.main(args + ['--rerun']) == 0
    assert len(works) == 2 and works[0] != works[1]
    assert all((work / 'console.log').read_text().startswith('progress') for work in works)
    row = json.loads((tmp_path / 'output/results/test-chip.json').read_text())['problems']['abs']
    assert row['work_dir'] == str(works[-1])
    assert row['status'] == ('failed' if exit_code else 'complete')
