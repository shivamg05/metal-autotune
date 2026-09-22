"""New starter ops reach real worker validation at registered benchmark shapes."""
import pytest

from autotuner.loop import JobRunner
from metalbench.bridge import problems, write_job


@pytest.mark.parametrize('name', ['instance_norm', 'group_norm', 'log_softmax', 'softmax_attention'])
def test_benchmark_fusion_starter_validates(tmp_path, name):
    problem = problems(('standard',))[name]
    manifest = write_job(problem, 1, 1, out=tmp_path / 'job')
    runner = JobRunner(manifest, tmp_path / 'work', judge_factory=lambda _: None)
    try:
        runner.load_model()
        runner.trace_workloads()
        regions = runner.build_regions()
        region = max(regions, key=lambda r: len(r.ops))
        required = {'instance_norm': 'mx.mean', 'group_norm': 'mx.mean',
                    'log_softmax': 'mx.logsumexp', 'softmax_attention': 'mx.softmax'}
        assert required[name] in region.ops, runner.report.stranded
        assert len(region.ops) >= (6 if 'norm' in name else 2)
        runner._capture([region])
        assert region.rejected is None
        # Validation only. The clock is disabled, so this value cannot decide a win.
        region.t_orig_ms[name] = region.t_rep_ms[name] = 1.0
        kernel = runner._build_scaffold(region)
        assert kernel.reference_sequence is None
        result = runner._evaluate_kernel(region, kernel, 'changing', run_clock=False)
        assert result.failed_gate is None, result
    finally:
        runner.tracer.uninstall()
