"""Keep checkpoint identity stable within a run; exports leave loading to the source."""
import json
from pathlib import Path

import pytest

from autotuner.artifact.bundle import (
    capture_checkpoint_loads, resolve_model_checkpoints, use_checkpoint_pins,
)


def checkpoint(path, content=b'weights'):
    path.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    (path / 'model.safetensors').write_bytes(content)
    (path / 'tokenizer.json').write_text('{}')
    return path


def model_source(tmp_path):
    model = tmp_path / 'model.py'
    model.write_text('from mlx_lm import load\ndef build():\n return load("org/model", revision="main")[0]\n')
    return model


def test_run_pins_come_from_actual_loads_not_source_literals(tmp_path):
    model = model_source(tmp_path)
    cache = checkpoint(tmp_path / 'cache' / 'commit-a')
    pins = resolve_model_checkpoints(model, captured={('org/model', 'main'): str(cache)})
    assert pins == [{'source': 'org/model', 'requested_revision': 'main',
                     'revision': 'commit-a', 'path': str(cache.resolve())}]


def test_missing_loaded_checkpoint_fails_before_search(tmp_path):
    with pytest.raises(ValueError, match='no longer available'):
        resolve_model_checkpoints(model_source(tmp_path),
                                  captured={('org/model', None): str(tmp_path / 'gone')})


def test_capture_records_helper_results_and_restores_on_failure(tmp_path, monkeypatch):
    import mlx_lm.utils as utils
    cache = checkpoint(tmp_path / 'snapshots' / 'commit-a')
    original = lambda *a, **kw: cache
    monkeypatch.setattr(utils, '_download', original)
    with pytest.raises(RuntimeError, match='builder failed'):
        with capture_checkpoint_loads() as captured:
            assert utils._download('org/model', revision='main') == cache
            assert captured == {('org/model', 'main'): str(cache.resolve())}
            raise RuntimeError('builder failed')
    assert utils._download is original


def test_capture_rejects_revision_change_between_baseline_builds(tmp_path, monkeypatch):
    import mlx_lm.utils as utils
    paths = iter((tmp_path / 'commit-a', tmp_path / 'commit-b'))
    original = lambda *a, **kw: next(paths)
    monkeypatch.setattr(utils, '_download', original)
    with capture_checkpoint_loads():
        utils._download('org/model')
        with pytest.raises(ValueError, match='revision changed'):
            utils._download('org/model')
    assert utils._download is original


def test_later_model_builds_use_pinned_local_files(tmp_path, monkeypatch):
    import mlx_lm.utils as utils
    cache = checkpoint(tmp_path / 'commit-a')
    calls = []
    def download(path, **kwargs):
        calls.append((path, kwargs))
        return Path(path)
    monkeypatch.setattr(utils, '_download', download)
    pins = [{'source': 'org/model', 'requested_revision': None,
             'path': str(cache), 'revision': 'commit-a'}]
    with use_checkpoint_pins(pins):
        assert utils._download('org/model') == cache
        with pytest.raises(ValueError, match='revision changed'):
            utils._download('org/model', revision='new-branch')
    assert calls == [(cache, {'revision': None, 'allow_patterns': None})]
    assert utils._download is download


def test_custom_models_do_not_require_mlx_lm_for_capture(monkeypatch):
    import builtins
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == 'mlx_lm.utils':
            raise ModuleNotFoundError('optional mlx_lm unavailable', name='mlx_lm')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', importing)
    with capture_checkpoint_loads() as captured:
        assert captured == {}
    with use_checkpoint_pins(None):
        pass
