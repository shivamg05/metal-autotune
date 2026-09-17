"""Mixed cache workloads reset real state and remain deployable."""
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import pytest

from autotuner.artifact.validate import check_outputs
from autotuner.loop import JobRunner
from autotuner.loop import RegionRun
from autotuner.ladder.gates import LadderResult
from autotuner.measure.session import Session
from autotuner_runtime.state import ContextStep, context_step, _cache_observation, _copy_cache
from autotuner_runtime.captured_kernels import capture_construction
from tests.fixtures.qwen_cache_model import build


@pytest.mark.parametrize('context,length', [(0, 4), (5, 3), (5, 1)])
def test_real_hybrid_cache_repeats_and_advances(context, length):
    # As in JobRunner, capture library kernel construction before imports.
    with capture_construction():
        model = build()
    prefix = mx.array([[i % 64 for i in range(context)]], dtype=mx.int32)
    x = mx.array([[i + 11 for i in range(length)]], dtype=mx.int32)
    step = context_step(model, context, prefix, [x])
    expected_cache = model.make_cache()
    if context:
        mx.eval(model(prefix, cache=expected_cache))
    initial = _cache_observation(step._cache)
    expected = model(x, cache=expected_cache)
    mx.eval(expected)
    for _ in range(3):
        actual = step(x)
        assert check_outputs(lambda: expected, lambda: actual, 'one')['passed']
        assert check_outputs(lambda: (x, initial),
                             lambda: (x, _cache_observation(step._cache)), 'reset')['passed']
    # Independent evolving sequence, including the recurrent state outputs.
    cache = _copy_cache(step._cache)
    outputs = []
    for _ in range(3):
        outputs.append(model(x, cache=cache))
        mx.eval(outputs[-1])
    expected = {'outputs': outputs, 'state': _cache_observation(cache)}
    assert check_outputs(lambda: expected,
                         lambda: step.sequence([x], 3, include_state=True), 'sequence')['passed']


def test_failed_recurrent_call_restores_values_metadata_and_aliases():
    class Cache:
        def __init__(self):
            self.data = mx.array([3.0])
            self.alias = self.data
            self.position = 4

    class Broken:
        def __call__(self, *, cache):
            cache[0].data[0] = 99.0
            cache[0].position = 100
            cache[0].extra = 'partial update'
            mx.eval(cache[0].data)
            raise RuntimeError('failed')

    cache = Cache()
    step = ContextStep(Broken(), [cache], 4)
    with pytest.raises(RuntimeError, match='failed'):
        step()
    assert cache.data.item() == 3 and cache.position == 4
    assert cache.alias is cache.data and not hasattr(cache, 'extra')


def test_sliding_cache_restores_overwritten_prefix():
    from mlx_lm.models.cache import RotatingKVCache

    class Model:
        def __call__(self, x, *, cache):
            return cache[0].update_and_fetch(x, x)[0]

    cache = RotatingKVCache(max_size=4, keep=0)
    mx.eval(cache.update_and_fetch(mx.ones((1, 1, 4, 8)), mx.ones((1, 1, 4, 8))))
    step = ContextStep(Model(), [cache], 4)
    before = _cache_observation([cache])
    x = mx.full((1, 1, 1, 8), 9.0)
    a, b = step(x), step(x)
    assert check_outputs(lambda: a, lambda: b, 'repeat')['passed']
    assert check_outputs(lambda: before, lambda: _cache_observation([cache]), 'reset')['passed']


def test_cache_only_gpu_work_remains_in_the_timed_graph(tmp_path):
    from tests.test_compiled_region_replay import assert_graph
    from types import SimpleNamespace

    kernel = mx.fast.metal_kernel(
        name='cache_only_write', input_names=['x'], output_names=['state'],
        source='uint i = thread_position_in_grid.x; state[i] = x[i] + 1.0f;')

    class Model:
        def __call__(self, x, *, cache):
            cache[0].value = kernel(inputs=[x], grid=(32, 1, 1), threadgroup=(32, 1, 1),
                                    output_shapes=[x.shape], output_dtypes=[x.dtype])[0]
            return {'logits': x, 'label': 'unchanged'}

    cache = SimpleNamespace(value=None)
    step = ContextStep(Model(), [cache], 0)
    outputs = step(mx.ones((32,)))
    assert cache.value is None
    assert outputs['label'] == 'unchanged'
    assert_graph(tmp_path / 'cache_work.dot', [outputs['logits']], 1)


@pytest.mark.parametrize('context', [0, 5])
def test_hybrid_cache_traces_reads_and_writes(tmp_path, context):
    model = Path(__file__).parent / 'fixtures/qwen_cache_model.py'
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(f'''model: {model}
baseline: plain
use_library_inference: false
workloads:
  - name: prompt
    context: {context}
    inputs: [{{shape: [1, 4], dtype: int32, high: 64}}]
''')
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda _: None,
                       session=Session(sleep=lambda _: None))
    try:
        runner.load_model()
        runner.trace_workloads()
        ops = [n.op for n in runner.traces['prompt'].nodes]
        assert 'state:ArraysCache.__getitem__' in ops
        assert ops.count('state:ArraysCache.__setitem__') == 2
        assert 'state:KVCache.update_and_fetch' in ops
        assert any(n.kernel_definition for n in runner.traces['prompt'].nodes)
        runner.build_regions()
    finally:
        runner.tracer.uninstall()


@pytest.mark.parametrize('fixture', ['qwen_cache_model.py', 'llama_cache_model.py'])
@pytest.mark.parametrize('context', [0, 5])
def test_cached_kernel_install_and_export_on_two_architectures(tmp_path, monkeypatch, fixture, context):
    model = Path(__file__).parent / 'fixtures' / fixture
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(f'''model: {model}
baseline: plain
use_library_inference: false
workloads:
  - name: prompt
    context: {context}
    inputs: [{{shape: [1, 4], dtype: int32, high: 64}}]
''')
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda _: None, clock_pairs=4,
                       session=Session(sleep=lambda _: None))
    # An identity seed is not a speedup. Only the speed decision is controlled;
    # execution, state correctness, installation and fresh export are real.
    monkeypatch.setattr(runner, '_model_win', lambda result: all(c.passed for c in result.checks))
    try:
        runner.load_model()
        runner.trace_workloads()
        regions = runner.build_regions()
        region = next(r for r in regions if r.ops == ('array.__add__',) and not r.rejected)
        kernel = runner._build_scaffold(region)
        region.p['prompt'] = 1.0
        region.t_orig_ms['prompt'] = region.t_rep_ms['prompt'] = 1.0
        runner.tracer.uninstall()
        assert runner._bind_and_promote(RegionRun(region), kernel,
            LadderResult('tentative_ship', None, {}, .1, 1., .1, 0., [])), runner.log.rows()[-1]
        assert runner.installed
        # The residual add lives in the block, which receives the cache: the
        # block compiles at the recorded position, empty or not. The hybrid's
        # recurrent block is not bitwise the same compiled, which identity
        # certification catches and answers with replay.
        installed = [r for r in runner.log.rows() if r['kind'] == 'installation'][-1]
        fallbacks = [r for r in runner.log.rows() if r['kind'] == 'graph_fallback']
        if fixture == 'llama_cache_model.py':
            assert installed['method'] == 'graph' and not fallbacks, (installed, fallbacks)
        else:
            assert installed['method'] == 'replay' and 'output changed' in fallbacks[-1]['reason'], fallbacks
        runner.final_ok = True
        artifact = runner._emit_artifact(tmp_path / 'artifact')
    finally:
        runner.tracer.uninstall()

    # Use only the exported model/runtime in a fresh process. Exercise a
    # caller-owned cache: empty prompt, populated prompt, then single tokens.
    code = '''
import importlib.util, sys
import mlx.core as mx
spec = importlib.util.spec_from_file_location('loader', sys.argv[1] + '/load.py')
loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
a = loader.load(patched=False).inference_model
b = loader.load().inference_model
from autotuner_runtime.state import _cache_observation
from autotuner_runtime.exact import bitwise_equal
from autotuner_runtime import kernels
from autotuner_runtime.swap import flatten_arrays
calls = []
original_call = kernels.call
def counted(*args, **kwargs):
    calls.append(args[0].kernel_id)
    return original_call(*args, **kwargs)
kernels.call = counted
ca, cb = a.make_cache(), b.make_cache()
context = int(sys.argv[2])
if context:
    # Different values from the harness's seeded prefix: cache contents must
    # be read at call time, never frozen into an installed wrapper.
    prefix = mx.array([[31 + i for i in range(context)]], dtype=mx.int32)
    mx.eval(a(prefix, cache=ca), b(prefix, cache=cb))
    calls.clear()
for index, length in enumerate((4, 4, 1, 1)):
    x = mx.array([list(range(1, length + 1))], dtype=mx.int32)
    ya, yb = a(x, cache=ca), b(x, cache=cb)
    mx.eval(ya, yb)
    assert bitwise_equal(ya, yb)
    sa, sb = _cache_observation(ca), _cache_observation(cb)
    assert all(bitwise_equal(x, y) for x, y in zip(flatten_arrays(sa), flatten_arrays(sb)))
    if index == 0:
        assert calls, 'cached prompt bypassed every installed kernel'
    calls.clear()
assert 'autotuner' not in sys.modules
print('cached inference passed')
'''
    proc = subprocess.run([sys.executable, '-c', code, str(artifact), str(context)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
