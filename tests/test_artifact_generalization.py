"""Portable sources and checkpoint paths retain their original meaning."""
import importlib.util
import json
from pathlib import Path
import sys

import mlx.core as mx
import pytest

from autotuner.artifact.bundle import ModelBundle, source_files, write_bundle
from autotuner.artifact.emit import emit_artifact, _CHECK
from autotuner.report import Report


@pytest.fixture(autouse=True)
def cpu_arrays():
    previous = mx.default_device()
    aliases = {"autotune_model", "artifact_apply", "artifact_load", "artifact_validate",
               "_artifact_apply", "_artifact_load"}
    saved_modules = {name: value for name, value in sys.modules.items()
                     if name in aliases or name.startswith("_artifact_model_")}
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)
        # These scripts normally run in separate interpreters. CPU tests run
        # them in-process, so restore their module registrations afterwards.
        for name in list(sys.modules):
            if name in aliases or name.startswith("_artifact_model_"):
                sys.modules.pop(name)
        sys.modules.update(saved_modules)


def module(path, name='artifact_test_load'):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_package_initializer_and_relative_attributes_travel(tmp_path):
    package = tmp_path / 'pkg'
    package.mkdir()
    (package / '__init__.py').write_text('VALUE = 42\n')
    model = package / 'model.py'
    model.write_text('from . import VALUE\ndef build():\n return lambda: VALUE\n')
    files, _, _ = source_files(model, tmp_path)
    assert package / '__init__.py' in files
    out = emit_artifact(tmp_path / 'artifact', [], [], Report(),
                        bundle=ModelBundle(model, baseline='plain', project_root=tmp_path))
    assert module(out / 'load.py').load(patched=False)() == 42


def test_declared_resource_keeps_internal_symlink_name(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'actual.txt').write_text('content')
    (tmp_path / 'alias.txt').symlink_to(data / 'actual.txt')
    model = tmp_path / 'model.py'
    model.write_text("from pathlib import Path\nARTIFACT_FILES = ['alias.txt']\n"
                     "def build():\n return lambda: Path(__file__).with_name('alias.txt').read_text()\n")
    out = emit_artifact(tmp_path / 'artifact', [], [], Report(),
                        bundle=ModelBundle(model, baseline='plain', project_root=tmp_path))
    assert (out / 'model' / 'alias.txt').is_file()
    assert not (out / 'model' / 'alias.txt').is_symlink()
    assert module(out / 'load.py').load(patched=False)() == 'content'


def test_declared_resource_cannot_follow_symlink_outside_project(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    outside = tmp_path / 'outside.txt'
    outside.write_text('not project data')
    (project / 'alias.txt').symlink_to(outside)
    model = project / 'model.py'
    model.write_text("ARTIFACT_FILES=['alias.txt']\ndef build(): pass\n")
    with pytest.raises(ValueError, match='inside the project'):
        source_files(model, project)


def fake_checkpoint(path, value):
    path.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    (path / 'tokenizer.json').write_text('{}')
    (path / 'model.safetensors').write_text(str(value))
    return path


def fake_loader(monkeypatch):
    import mlx_lm
    import mlx_lm.utils as utils
    def download(path, **kwargs):
        path = Path(path)
        assert path.is_absolute(), 'must use the pinned files, never resolve remote main'
        return path
    def load(source):
        path = utils._download(source)
        value = float((path / 'model.safetensors').read_text())
        return (lambda x: x + value), None
    monkeypatch.setattr(utils, '_download', download)
    monkeypatch.setattr(mlx_lm, 'load', load)


def test_loader_follows_source_without_copying_or_redirecting_weights(tmp_path, monkeypatch):
    cache = fake_checkpoint(tmp_path / 'cache' / 'commit-a', 2)
    model = tmp_path / 'model.py'
    model.write_text('from mlx_lm import load\nMODEL=load("org/model")[0]\ndef build():\n return MODEL\n')
    pins = [{'source': 'org/model', 'requested_revision': None,
             'revision': 'commit-a', 'path': str(cache)}]
    out = emit_artifact(tmp_path / 'recovery', [], [], Report(),
                        bundle=ModelBundle(model, baseline='plain', project_root=tmp_path,
                                           recovery=True, checkpoint_pins=pins))
    metadata = json.loads((out / 'bundle.json').read_text())
    assert metadata['weight_policy'] == 'model_source'
    assert not metadata['checkpoint_resources_included']
    assert not (out / 'model' / '_checkpoints').exists()
    fake_loader(monkeypatch)
    import mlx_lm.utils as utils
    monkeypatch.setattr(utils, '_download', lambda source, **kw: cache)
    loaded = module(out / 'load.py').load(patched=False)
    assert loaded(mx.array([1.0])).item() == 3.0


def test_fresh_apply_validation_uses_current_source_weights(tmp_path, monkeypatch):
    cache = fake_checkpoint(tmp_path / 'cache' / 'commit-a', 2)
    model = tmp_path / 'model.py'
    model.write_text('from mlx_lm import load\nMODEL=load("org/model")[0]\ndef build():\n return MODEL\n')
    pins = [{'source': 'org/model', 'requested_revision': None,
             'revision': 'commit-a', 'path': str(cache)}]
    out = emit_artifact(tmp_path / 'artifact', [], [], Report(),
                        bundle=ModelBundle(model, baseline='plain', project_root=tmp_path, checkpoint_pins=pins))
    # The cache changes. Both fresh validation arms must use the current weights.
    (cache / 'model.safetensors').write_text('999')
    fake_loader(monkeypatch)
    import mlx_lm.utils as utils
    monkeypatch.setattr(utils, '_download', lambda source, **kw: cache)
    inputs, expected = tmp_path / 'inputs.safetensors', tmp_path / 'expected.safetensors'
    mx.save_safetensors(str(inputs), {'i0': mx.array([1.0])})
    mx.save_safetensors(str(expected), {'o0': mx.array([3.0])})
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps({'context': None, 'cases': [
        {'name': 'main', 'inputs': str(inputs), 'expected': str(expected)}]}))
    monkeypatch.setattr(sys, 'argv', ['check', str(model), str(out / 'apply.py'), str(cases), 'apply'])
    exec(compile(_CHECK, '<artifact-check>', 'exec'), {})


@pytest.mark.parametrize('call', [
    'load("org/model", None, None, None, False, False, "commit-a")',
    'load("org/model", adapter_path="/external/adapter")',
    'load(**options)', 'load(MODEL)', 'getattr(mlx_lm, "load")(MODEL)',
])
def test_builders_are_copied_without_interpreting_checkpoint_arguments(tmp_path, call):
    model = tmp_path / 'model.py'
    model.write_text('import mlx_lm\nfrom mlx_lm import load\n'
                     f'def build():\n return {call}[0]\n')
    out = emit_artifact(tmp_path / 'artifact', [], [], Report(),
                        bundle=ModelBundle(model, baseline='plain', project_root=tmp_path))
    assert (out / 'model/model.py').read_bytes() == model.read_bytes()
    assert not list((out / 'model').rglob('*.safetensors'))


def test_dynamic_actual_loads_are_recorded_for_run_rebuilds(tmp_path):
    from autotuner.artifact.bundle import resolve_model_checkpoints
    model = tmp_path / 'model.py'
    model.write_text('import mlx_lm\ndef build():\n return getattr(mlx_lm,"load")("org/model")[0]\n')
    cached = tmp_path / 'cached-commit'
    cached.mkdir()
    pins = resolve_model_checkpoints(model, captured={('org/model',None):str(cached)})
    assert pins[0]['path'] == str(cached) and pins[0]['source'] == 'org/model'
