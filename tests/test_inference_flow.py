"""Inference measurement and graph patches survive the whole delivery path.

These are small real architectures, not headline performance benchmarks. The
baseline is untouched native generation on identical weights, prompt and cache.
"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import mlx.core as mx
import pytest
import yaml

from autotuner.artifact.bundle import ModelBundle
from autotuner.artifact.emit import emit_artifact
from autotuner.bind.emit import Splice
from autotuner.bind.graph import emit_graph_wrapper_variants
from autotuner.loop import JobRunner
from autotuner.measure.session import Session
from autotuner.report import Report
from autotuner_runtime.inference import LibraryInference
from autotuner_runtime.kernels import KernelSpec
from autotuner_runtime.state import correctness_call


FIXTURES = Path(__file__).parent / "fixtures"


def _job(tmp_path, model, *, context=None, enabled=None, workloads=None):
    workload = {"name": "banana", "inputs": [{"shape": [1, 4], "dtype": "int32", "high": 64}]}
    if context is not None:
        workload["context"] = context
    config = {"model": str(model), "baseline": "compiled", "workloads": workloads or [workload],
              "final_benchmark": {"steps": 3, "pairs": 4, "warmup_steps": 1}}
    if enabled is not None:
        config["use_library_inference"] = enabled
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(config))
    return JobRunner(manifest, tmp_path / "work", judge_factory=lambda _: None,
                     session=Session(sleep=lambda _: None))


@pytest.mark.parametrize("kind", ["llama", "qwen"])
@pytest.mark.parametrize("context", [None, 0, 5])
def test_job_uses_the_same_complete_inference_task_for_trace_and_timing(tmp_path, kind, context):
    # load_model installs capture before importing the architecture. A subprocess
    # also isolates native custom definitions constructed by earlier tests.
    script = textwrap.dedent("""
        import sys
        from pathlib import Path
        from tests.test_inference_flow import _job, FIXTURES
        from autotuner_runtime.inference import LibraryInference
        from autotuner_runtime.state import correctness_call
        from autotuner.artifact.validate import check_outputs
        root, kind, context = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
        context = None if context == 'None' else int(context)
        runner = _job(root, FIXTURES / (kind + '_cache_model.py'), context=context)
        try:
            runner.load_model()
            assert runner.use_library_inference is True
            assert isinstance(runner.model, LibraryInference)
            runner.trace_workloads()
            assert len(runner.traces['banana'].step_outputs) == 3
            runner.tracer.uninstall()
            inputs = runner.tensors['banana']
            expected = correctness_call(runner.baseline_model, inputs)
            assert len(expected['tokens']) == 3 and 'state' in expected
            for arm in runner._timed_arms()['banana']:
                assert len(arm()['tokens']) == 3
            check = check_outputs(lambda: expected,
                lambda: correctness_call(runner.model, inputs), 'banana')
            assert check['passed'], check
            assert runner.report.constants['measurement']['kind'] == 'library_generation'
        finally:
            runner.tracer.uninstall()
    """)
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), kind, str(context)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_multiple_workloads_keep_separate_native_inference_arms(tmp_path):
    workloads = [{"name": name, "inputs": [{"shape": [1, size], "dtype": "int32", "high": 64}]}
                 for name, size in (("short", 1), ("long", 7))]
    runner = _job(tmp_path, FIXTURES / "llama_cache_model.py", workloads=workloads)
    try:
        runner.load_model()
        runner.trace_workloads()
        runner.tracer.uninstall()
        arms = runner._timed_arms()
        assert set(arms) == {"short", "long"}
        for name, pair in arms.items():
            assert runner.tensors[name][0].shape[1] == (1 if name == "short" else 7)
            assert all(len(run()["tokens"]) == 3 for run in pair)
        assert len(runner.traces["short"].nodes) != len(runner.traces["long"].nodes)
    finally:
        runner.tracer.uninstall()


@pytest.mark.parametrize("context", [None, 0, 5])
def test_final_confirmation_runs_one_generation_task_per_arm(tmp_path, monkeypatch, context):
    runner = _job(tmp_path, FIXTURES / "llama_cache_model.py", context=context)
    try:
        runner.load_model()
        runner.trace_workloads()
        runner.tracer.uninstall()
        runner.baseline = "plain"
        seen = []

        def compare(session, baseline, candidate, warm_base, warm_candidate, **kwargs):
            for run in (baseline, candidate, warm_base, warm_candidate):
                result = run()
                assert isinstance(result, dict) and len(result["tokens"]) == 3
                seen.append(result["tokens"])
            return {"baseline_sequence_ms": 2.0, "candidate_sequence_ms": 1.9,
                    "timing": {"baseline_ms": [2.0] * 4, "candidate_ms": [1.9] * 4}}

        monkeypatch.setattr("autotuner.loop.compare_sequences", compare)
        rows, result = runner._final_sequences([])
        assert len(seen) == 4 and len(result.checks) == 1
        assert rows["banana"]["workload_kind"] == "library_generation"
        assert rows["banana"]["steps"] == 3
    finally:
        runner.tracer.uninstall()


def test_explicit_forward_preserves_a_language_models_forward_call(tmp_path):
    runner = _job(tmp_path, FIXTURES / "llama_cache_model.py", enabled=False)
    try:
        runner.load_model()
        runner.trace_workloads()
        runner.tracer.uninstall()
        assert not runner.use_library_inference
        output = runner.model(*runner.tensors["banana"])
        assert output.shape == (1, 4, 64)
    finally:
        runner.tracer.uninstall()


def test_denoiser_stays_forward_and_explicit_inference_fails_early(tmp_path):
    # Use the same forward contract as a denoiser: image, conditioning, timestep.
    model = tmp_path / "denoiser.py"
    model.write_text("import mlx.core as mx\n"
                     "def build():\n"
                     "    return lambda image, condition, time: mx.tanh(image + condition * time)\n")
    workloads = [{"name": "arbitrary", "inputs": [{"shape": [1, 4], "dtype": "float32"}] * 3}]
    runner = _job(tmp_path, model, workloads=workloads)
    try:
        runner.load_model()
        assert not runner.use_library_inference
    finally:
        runner.tracer.uninstall()
    other = tmp_path / "explicit"
    other.mkdir()
    runner = _job(other, model, workloads=workloads, enabled=True)
    with pytest.raises(ValueError, match="unsupported"):
        runner.load_model()
    assert not runner.traces and runner.tracer.verify_restored() == []


def test_graph_kernel_bundle_loads_and_runs_native_inference_without_harness(tmp_path):
    # An identity operation on a real language model provides a known-correct
    # replacement. Its purpose is installation and packaging, not a speed claim.
    source = (FIXTURES / "llama_cache_model.py").read_text()
    source += textwrap.dedent("""
        _build = build
        from mlx_lm.models.llama import Model as NativeModel
        class Probe(nn.Module):
            def __call__(self, value):
                return mx.multiply(value, 1.0)
        class Model(NativeModel):
            def __init__(self, args):
                super().__init__(args)
                self.probe = Probe()
            def __call__(self, tokens, cache=None):
                return self.probe(super().__call__(tokens, cache=cache))
        def build():
            original = _build()
            model = Model(original.args)
            nn.quantize(model, group_size=64, bits=4)
            model.update(original.parameters())
            model.eval()
            return model
    """)
    model_file = tmp_path / "model.py"
    model_file.write_text(source)
    runner = _job(tmp_path, model_file, context=5)
    try:
        runner.load_model()
        runner.trace_workloads()
        runner.tracer.uninstall()
        trace = runner.traces["banana"]
        kernel = KernelSpec(kernel_id="copy", name="flow_copy", input_names=("x",),
            output_names=("out",), source="uint i = thread_position_in_grid.x; out[i] = x[i];",
            grid=("in0.shape[0] * in0.shape[1] * in0.shape[2]", "1", "1"),
            threadgroup=("64", "1", "1"),
            output_shapes=(("in0.shape[0]", "in0.shape[1]", "in0.shape[2]"),),
            output_dtypes=("float32",))
        variants = []
        for scope in trace.scope_calls:
            if scope.address.startswith("model.probe@"):
                node = next(n for n in trace.nodes if n.module_address == scope.address and n.op == "mx.multiply")
                variants.append((trace, scope, [Splice(kernel=kernel, start_seq=node.seq,
                    end_seq=node.seq, input_ids=node.in_arrays, output_ids=node.out_arrays)]))
        assert variants
        wrapper = emit_graph_wrapper_variants(variants, "InferenceGraphProbe")
        bundle = ModelBundle(model_path=model_file, project_root=tmp_path, baseline="plain",
            workloads=runner.tensors, workload_config=runner.manifest.workloads,
            final_benchmark={"steps": 3, "pairs": 4, "warmup_steps": 1},
            use_library_inference=True,
            context={"workload": "banana", "context": 5, "seed": runner.context_seed,
                     "tokens": runner.context_tokens})
        out = emit_artifact(tmp_path / "artifact", [kernel], [wrapper], Report(), bundle=bundle)
    finally:
        runner.tracer.uninstall()
    metadata = json.loads((out / "bundle.json").read_text())
    assert metadata["measurement"] == {"kind": "library_generation", "generated_tokens": 3}
    assert not metadata["checkpoint_resources_included"]
    child = textwrap.dedent("""
        import importlib.util, sys
        from pathlib import Path
        import mlx.core as mx
        art = Path(sys.argv[1])
        def module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            value = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(value)
            return value
        loader = module('bundle_load', art / 'load.py')
        original = loader.load(patched=False)
        patched = loader.load(share_weights_with=original.model)
        validator = module('bundle_validate', art / 'validate.py')
        rows = validator.validate(original, patched, include_sequences=True)
        assert all(row['passed'] for row in rows), rows
        tokens = mx.load(str(art / 'workloads/banana.safetensors'))['i0']
        assert len(patched(tokens)['tokens']) == 3
        assert type(patched.inference_model.probe).__mro__[1].__name__ == 'GraphWrapper'
        assert len(loader.load(generated_tokens=2)(tokens)['tokens']) == 2
        bare = original.definition.build()
        bare.update(original.inference_model.parameters())
        bare = module('bundle_apply', art / 'apply.py').apply(bare)
        from autotuner_runtime.inference import LibraryInference
        from autotuner_runtime.state import correctness_call
        prefix = mx.load(str(art / 'workloads/banana.context.safetensors'))['tokens']
        check = validator.check_outputs(
            lambda: correctness_call(original, [tokens]),
            lambda: LibraryInference(bare, steps=3, prefix_tokens=prefix).correctness(tokens), 'bare apply')
        assert check['passed'], check
        assert 'autotuner' not in sys.modules
    """)
    result = subprocess.run([sys.executable, "-c", child, str(out)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
