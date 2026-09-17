"""Native custom kernels survive serialization, model ownership and process exit."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mlx.core as mx
import mlx.nn as nn
import pytest

from autotuner.trace import Tracer
from autotuner.trace.serialize import nodes_to_json, nodes_from_json
from autotuner.trace.replay import replay, prepare_replay
from autotuner.bind.emit import MODULE_HEADER, emit_wrapper
from autotuner.ladder.golden import fp32_reference


def make_kernel():
    return mx.fast.metal_kernel(
        name="portable_scale", input_names=["x", "n"], output_names=["y", "z"],
        source="uint i = thread_position_in_grid.x; y[i] = twice(x[i]); z[i] = x[i] + n;",
        header="template <typename T> T twice(T x) { return x + x; }",
        ensure_row_contiguous=True, atomic_outputs=False,
        compile_options={"math_mode": "safe"})


class Model(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        # Deliberately held only by this instance, with no importable name.
        self.kernel = kernel

    def __call__(self, x):
        return self.kernel(inputs=[x, 3], grid=(4, 1, 1), threadgroup=(4, 1, 1),
                           output_shapes=[(4,), (4,)], output_dtypes=[x.dtype, x.dtype],
                           init_value=0)


def test_definition_launch_and_sequential_sessions(tmp_path):
    x = mx.arange(4, dtype=mx.float32)
    t = Tracer(); t.install()
    try:
        model = Model(make_kernel())
        first, expected = t.trace(model, [x])
    finally:
        t.uninstall()
    # A retained callable must follow the active tracer, not the one that built it.
    t = Tracer(); t.install()
    try:
        trace, actual = t.trace(model, [x])
        assert nodes_to_json(first.nodes) == nodes_to_json(trace.nodes)
        nodes = nodes_from_json(nodes_to_json(trace.nodes))
        assert len(nodes) == 1
        assert nodes[0].kernel_definition['kwargs']['compile_options'] == {'math_mode': 'safe'}
        assert nodes[0].scalar_args['kwargs']['init_value'] == 0
        binds = {next(iter(trace.inputs)): x}
        for out in (list(replay(nodes, binds, trace.step_outputs).values()),
                    prepare_replay(nodes, binds, trace.step_outputs)(binds), actual):
            assert all(mx.array_equal(a, b).item() for a, b in zip(out, expected))
        emitted = emit_wrapper(trace, trace.scope_calls[0], [], 'Portable')
    finally:
        t.uninstall()
    # Only the runtime and generated wrapper travel. No original model/module.
    runtime = Path(__file__).resolve().parents[1] / 'autotuner_runtime'
    shutil.copytree(runtime, tmp_path / 'autotuner_runtime', ignore=shutil.ignore_patterns('__pycache__'))
    (tmp_path / 'wrapper.py').write_text(MODULE_HEADER + emitted.source)
    script = '''
import sys
sys.path.insert(0, sys.argv[1])
import mlx.core as mx
import mlx.nn as nn
from wrapper import Portable
class Original(nn.Module):
    def __call__(self, x): raise AssertionError("unexpected fallback")
y, z = Portable(Original())(mx.arange(4, dtype=mx.float32))
mx.eval(y, z)
assert y.tolist() == [0, 2, 4, 6] and z.tolist() == [3, 4, 5, 6]
assert not any(k == 'autotuner' or k.startswith('autotuner.') for k in sys.modules)
'''
    subprocess.run([sys.executable, '-I', '-c', script, str(tmp_path)], check=True,
                   capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize('hidden_by_add', [False, True])
def test_preexisting_raw_kernel_cannot_masquerade_as_fp32(hidden_by_add):
    # Constructed before interception, so its fp16 arithmetic is invisible.
    kernel = mx.fast.metal_kernel(name='hidden_half', input_names=['x'], output_names=['y'],
                                 source='y[0] = float(half(x[0]));')
    class Hidden(nn.Module):
        def __call__(self, x):
            y = kernel(inputs=[x], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                       output_shapes=[(1,)], output_dtypes=[mx.float32])[0]
            return y + 1 if hidden_by_add else y
    t = Tracer(); t.install()
    try:
        with pytest.raises(ValueError, match='unrecorded operation'):
            fp32_reference(Hidden(), [mx.array([0.10001])], t, check_graph=False)
    finally:
        t.uninstall()


def test_retrace_checks_native_definition_and_launch():
    from dataclasses import replace
    from autotuner.bind.verify import verify_retrace
    from autotuner.ladder.golden import golden_outputs
    t = Tracer(); t.install()
    try:
        x = mx.arange(4, dtype=mx.float32)
        trace, _ = t.trace(Model(make_kernel()), [x])
        node = trace.nodes[0]
        changed = dict(node.kernel_definition, kwargs=dict(node.kernel_definition['kwargs'],
                                                         source='y[0] = 0; z[0] = 0;'))
        other = replace(trace, nodes=(replace(node, kernel_definition=changed),))
        assert not verify_retrace(trace, other, [], []).ok
        launch = dict(node.scalar_args, kwargs=dict(node.scalar_args['kwargs'], init_value=1))
        other = replace(trace, nodes=(replace(node, scalar_args=launch),))
        assert not verify_retrace(trace, other, [], []).ok
        with pytest.raises(ValueError, match='no audited fp32 reference'):
            golden_outputs(trace.nodes, {next(iter(trace.inputs)): x}, trace.step_outputs)
    finally:
        t.uninstall()


def test_failed_install_releases_recorder(monkeypatch):
    t = Tracer()
    original = mx.fast.metal_kernel
    def fail():
        raise RuntimeError('injected install failure')
    monkeypatch.setattr(t.patcher, '_patch_precompiled', fail)
    with pytest.raises(RuntimeError, match='injected'):
        t.install()
    assert mx.fast.metal_kernel is original
    other = Tracer(); other.install()
    try:
        with pytest.raises(RuntimeError, match='already active'):
            Tracer().install()
        trace, _ = other.trace(Model(make_kernel()), [mx.arange(4, dtype=mx.float32)])
        assert trace.nodes[0].kernel_definition is not None
    finally:
        other.uninstall()
