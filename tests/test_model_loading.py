"""Users supply a plain build(); the runner owns kernel capture ordering."""
import textwrap

import mlx.core as mx
import pytest

from autotuner.loop import JobRunner
from autotuner.measure.session import Session
from autotuner.trace.replay import replay
from autotuner.trace.serialize import nodes_from_json, nodes_to_json


@pytest.mark.parametrize('at_import', [True, False])
def test_runner_captures_initial_and_rebuilt_model(tmp_path, at_import):
    factory = '''mx.fast.metal_kernel(name="load_scale", input_names=["x"],
        output_names=["y"], source="uint i = thread_position_in_grid.x; y[i] = x[i] * 2;")'''
    source = 'import mlx.core as mx\nimport mlx.nn as nn\n'
    if at_import:
        source += 'kernel = ' + factory + '\n'
    source += 'class Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n'
    source += '        self.kernel = ' + ('kernel' if at_import else factory) + '\n'
    source += '''    def __call__(self, x):
        return self.kernel(inputs=[x], grid=(4,1,1), threadgroup=(4,1,1),
                           output_shapes=[(4,)], output_dtypes=[x.dtype])[0]
def build():
    return Model()
'''
    (tmp_path / 'model.py').write_text(source)
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(textwrap.dedent('''
        model: model.py
        workloads:
          - name: main
            inputs: [{shape: [4], dtype: float32}]
        budget: {per_region: 1, total: 1}
    '''))
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda region: None,
                       session=Session(sleep=lambda s: None))
    original_factory = mx.fast.metal_kernel
    try:
        runner.load_model()
        runner.trace_workloads()
        initial = runner.traces['main']
        assert len(initial.nodes) == 1 and initial.nodes[0].kernel_definition is not None
        runner.tracer.uninstall()
        rebuilt = runner.build_model()
        assert mx.fast.metal_kernel is original_factory
        runner.tracer.install()
        inputs = runner.tensors['main']
        trace, out = runner.tracer.trace(rebuilt, inputs)
        nodes = nodes_from_json(nodes_to_json(trace.nodes))
        assert nodes[0].kernel_definition == initial.nodes[0].kernel_definition
        got = replay(nodes, {next(iter(trace.inputs)): inputs[0]}, trace.step_outputs)
        assert mx.array_equal(got[trace.step_outputs[0]], out).item()
        runner.tracer.uninstall()
        build = runner.model_module.build
        def failed_build():
            build()
            raise RuntimeError("injected build failure")
        runner.model_module.build = failed_build
        with pytest.raises(RuntimeError, match="injected build failure"):
            runner.build_model()
        assert mx.fast.metal_kernel is original_factory
    finally:
        runner.tracer.uninstall()
    assert mx.fast.metal_kernel is original_factory
