"""Cross-model failures: multi-output wiring and independently named kernels."""
from types import SimpleNamespace

from autotuner.loop import JobRunner, _ops_view
from autotuner.regions.types import Region, Stretch
from autotuner.trace.recorder import ArrayRef
from autotuner_runtime.kernels import KernelSpec


def test_judge_wiring_distinguishes_every_intermediate_output():
    nodes = [SimpleNamespace(seq=0, op='metal_kernel', in_arrays=(1,), out_arrays=(2, 3),
                             scalar_args={'args': (ArrayRef(0),), 'kwargs': {}}),
             SimpleNamespace(seq=1, op='mx.subtract', in_arrays=(2, 3), out_arrays=(4,),
                             scalar_args={'args': (ArrayRef(0), ArrayRef(1)), 'kwargs': {}})]
    span = Stretch('main', 0, 1, (1,), (4,), ())
    view = _ops_view(SimpleNamespace(nodes=nodes), span)
    assert view[0]['outputs'][0] != view[0]['outputs'][1]
    assert view[1]['args'] == view[0]['outputs']
    assert view[1]['outputs'] == ['out0']


def test_regions_sharing_display_prefix_cannot_overwrite_kernel_files(tmp_path):
    from autotuner.artifact.emit import write_kernel
    runner = object.__new__(JobRunner)
    seed = KernelSpec('seed', 'seed', ('in0',), ('out0',), 'out0[0]=in0[0];')
    regions = [Region('abcdef0000000000', ('mx.exp',)), Region('abcdef1111111111', ('mx.sin',))]
    specs = [runner._rename(seed, region, 'h1') for region in regions]
    assert specs[0].kernel_id != specs[1].kernel_id
    for spec in specs:
        write_kernel(tmp_path, spec)
    assert len(list(tmp_path.glob('*.metal'))) == 2


def test_checkpoint_capture_includes_model_import(tmp_path, monkeypatch):
    import mlx.core as mx
    import mlx_lm
    import mlx_lm.utils as utils
    from autotuner.artifact import bundle
    from autotuner.measure.session import Session
    source = tmp_path / 'model.py'
    source.write_text('from mlx_lm import load\nmodel, _ = load("org/model")\ndef build():\n return model\n')
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text('model: model.py\nworkloads:\n - name: main\n   inputs: [{shape: [4], dtype: float32}]\nbudget: {per_region: 1, total: 1}\n')
    cached = tmp_path / 'loaded-commit'
    cached.mkdir()
    monkeypatch.setattr(utils, '_download', lambda *a, **kw: cached)
    def load(path):
        utils._download(path)
        return (lambda x: x), None
    monkeypatch.setattr(mlx_lm, 'load', load)
    observed = {}
    def pins(model_path, *, captured):
        observed.update(captured)
        return []
    monkeypatch.setattr(bundle, 'resolve_model_checkpoints', pins)
    monkeypatch.setattr('autotuner.manifest.check_build', lambda manifest: None)
    monkeypatch.setattr(mx, 'clear_cache', lambda: None)
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda r: None,
                       session=Session(sleep=lambda s: None))
    try:
        runner.load_model()
        assert observed == {('org/model', None): str(cached.resolve())}
    finally:
        runner.tracer.uninstall()


def test_shared_layers_cannot_contaminate_the_baseline():
    import mlx.nn as nn
    import pytest
    from autotuner_runtime.swap import require_independent_models
    shared = nn.ReLU()
    left, right = nn.Sequential(shared), nn.Sequential(shared)
    with pytest.raises(ValueError, match='fresh model and layer instances'):
        require_independent_models(left, right)
    with pytest.raises(ValueError, match='<root>'):
        require_independent_models(left, left)
    require_independent_models(left, nn.Sequential(nn.ReLU()))
    import mlx.core as mx
    a, b = nn.Module(), nn.Module()
    a.weight = b.weight = mx.array([1.0])
    require_independent_models(a, b)  # immutable weight sharing remains supported


def test_runner_rejects_cached_model_before_search(tmp_path):
    import pytest
    from autotuner.measure.session import Session
    source = tmp_path / 'model.py'
    source.write_text('import mlx.nn as nn\nlayer = nn.ReLU()\ndef build():\n return nn.Sequential(layer)\n')
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text('model: model.py\nworkloads:\n - name: main\n   inputs: [{shape: [4], dtype: float32}]\n')
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda r: None,
                       session=Session(sleep=lambda s: None))
    with pytest.raises(ValueError, match='build.*reused a model module'):
        runner.load_model()
    assert not runner.tracer.patcher.installed


def test_boundary_labels_cannot_alias_or_escape_the_run(tmp_path):
    from autotuner.regions.store import BoundaryStore
    store = BoundaryStore(tmp_path)
    labels = ['prefill/512', 'prefill%2F512', '../decode', '.', '..', '/decode', 'decode']
    paths = [store._path('abcdef', label, 0, 'inputs') for label in labels]
    assert len(set(paths)) == len(labels)
    for label, path in zip(labels, paths):
        assert path.resolve().is_relative_to(tmp_path / 'abcdef')
        path.parent.mkdir(parents=True)
        path.touch()
        assert store.set_count('abcdef', label) == 1


def test_user_workload_names_cannot_alias_generated_shape_labels(tmp_path):
    import pytest
    from autotuner.manifest import ManifestError, load
    (tmp_path / 'model.py').write_text('def build():\n return lambda x: x\n')
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text('model: model.py\nworkloads:\n - name: main@L=4\n   inputs: [{shape: [4], dtype: float32}]\n')
    with pytest.raises(ManifestError, match='reserved for generated shape labels'):
        load(manifest)


def test_compile_survival_check_uses_bits_for_nan_and_signed_zero():
    import mlx.core as mx
    import pytest
    runner = object.__new__(JobRunner)
    before = mx.array([float('nan'), 0.0])
    runner.model = lambda: before
    runner._assert_survived_compile('main', [], [before])
    runner.model = lambda: mx.array([float('nan'), -0.0])
    with pytest.raises(RuntimeError, match='compiled baseline changed'):
        runner._assert_survived_compile('main', [], [before])
