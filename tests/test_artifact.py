"""The artifact loads in a fresh process with no harness import and
reproduces the patched model's outputs; with a bundle it also carries the
model's source and inputs, and its load, validate and benchmark scripts run
there on their own."""

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import mlx.core as mx

from autotuner.artifact.bundle import ModelBundle
from autotuner.artifact.emit import check_apply_many, dedupe_wrappers, emit_artifact
from autotuner.bind.emit import EmittedWrapper, Splice, emit_wrapper
from autotuner.loop import _load_class
from autotuner.manifest import InputSpec, Workload
from autotuner.report import Report
from autotuner.trace import Tracer
from autotuner_runtime.kernels import KernelSpec
from autotuner_runtime.swap import install

FIXTURES = Path(__file__).parent / "fixtures"

ADD_KERNEL = KernelSpec(
    kernel_id="k_art_add", name="artifact_test_add",
    input_names=("a", "b"), output_names=("out",),
    source="uint i = thread_position_in_grid.x;\nout[i] = a[i] + b[i];",
    grid=("in0.shape[0] * in0.shape[1]", "1", "1"),
    threadgroup=("min(in0.shape[0] * in0.shape[1], 256)", "1", "1"),
    output_shapes=(("in0.shape[0]", "in0.shape[1]"),),
    output_dtypes=("float32",),
)


def _fixture_model(name):
    spec = importlib.util.spec_from_file_location("fx", FIXTURES / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def _emit_layer(model, x, address, class_name):
    """The residual add of one repeated_layers layer, cut to ADD_KERNEL."""
    tracer = Tracer()
    tracer.install()
    try:
        trace, _ = tracer.trace(model, [x])
    finally:
        tracer.uninstall()
    add_node = next(n for n in trace.nodes if n.op == "array.__add__" and n.module_address == address)
    splice = Splice(kernel=ADD_KERNEL, start_seq=add_node.seq, end_seq=add_node.seq,
                    input_ids=tuple(add_node.in_arrays), output_ids=tuple(add_node.out_arrays))
    scope = next(sc for sc in trace.scope_calls if sc.address == address)
    return emit_wrapper(trace, scope, [splice], class_name)


def test_artifact_applies_in_fresh_process(tmp_path):
    model = _fixture_model("repeated_layers.py")
    x = mx.random.normal((4, 16), key=mx.random.key(0))
    emitted = _emit_layer(model, x, "layers.3@0", "ArtLayer3")

    out = emit_artifact(tmp_path / "artifact", [ADD_KERNEL], [emitted], Report())
    assert (out / "kernels" / "k_art_add.metal").exists()
    assert (out / "patch" / "wrappers.py").exists()
    assert (out / "runtime" / "autotuner_runtime" / "kernels.py").exists()
    assert (out / "runtime" / "autotuner_runtime" / "stats.py").exists()
    assert (out / "buffers").is_dir()
    assert not (out / "load.py").exists()  # no bundle was given

    baseline = model(x)
    mx.eval(baseline)

    child = textwrap.dedent(f"""
        import importlib.util, sys
        from pathlib import Path
        import mlx.core as mx

        art = Path({str(out)!r})
        sys.path.insert(0, str(art.parent))
        for mod_name in list(sys.modules):
            assert not mod_name.startswith("autotuner"), mod_name

        spec = importlib.util.spec_from_file_location("fx", {str(FIXTURES / "repeated_layers.py")!r})
        fx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fx)
        model = fx.build()

        apply_spec = importlib.util.spec_from_file_location("art_apply", art / "apply.py")
        art_apply = importlib.util.module_from_spec(apply_spec)
        apply_spec.loader.exec_module(art_apply)
        patched = art_apply.apply(model)

        assert "autotuner" not in sys.modules, "the artifact must not import the harness"
        x = mx.random.normal((4, 16), key=mx.random.key(0))
        y = patched(x)
        mx.eval(y)
        print("SHAPE", y.shape)
        print("SUM", float(y.astype(mx.float32).sum()))
    """)
    proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    lines = dict(l.split(" ", 1) for l in proc.stdout.strip().splitlines())
    assert lines["SHAPE"] == "(4, 16)"
    assert abs(float(lines["SUM"]) - float(baseline.astype(mx.float32).sum())) < 1e-3


def _wrapper(class_name, scope_path, body, kernel_ids):
    source = f"\n\nclass {class_name}(ReplayWrapper):\n    KERNEL_IDS = {kernel_ids!r}\n\n{body}"
    return EmittedWrapper(class_name, scope_path, source, [], list(kernel_ids))


def test_dedupe_shares_identical_bodies_and_keeps_distinct_ones_apart():
    same = "    def __call__(self, a0):\n        return a0\n"
    other = "    def __call__(self, a0):\n        return a0 + 1\n"
    wrappers = [
        _wrapper("W_layers_0", "layers.0", same, ["r1_h2"]),
        _wrapper("W_layers_1", "layers.1", same, ["r1_h2"]),
        _wrapper("W_layers_2", "layers.2", other, ["r1_h2"]),        # same kernel, different body
        _wrapper("W_head", "head", same.replace("a0", "b0"), ["k-a", "k-a", "k.b"]),
        EmittedWrapper("Saved", "layer", "class Saved: pass\n", [], ["accepted"]),  # not a class line
    ]
    module, table = dedupe_wrappers(wrappers)
    classes = [line for line in module.splitlines() if line.startswith("class ")]
    assert classes == ["class W_r1_h2(ReplayWrapper):", "class W_r1_h2_2(ReplayWrapper):",
                       "class W_k_a__k_b(ReplayWrapper):", "class Saved: pass"]
    assert "# serves layers.0, layers.1\nclass W_r1_h2(" in module
    assert "# serves layers.2\nclass W_r1_h2_2(" in module
    assert module.count(same) == 1 and other in module and "class Saved: pass\n" in module
    assert [(row["scope_path"], row["wrapper_class"]) for row in table] == [
        ("layers.0", "W_r1_h2"), ("layers.1", "W_r1_h2"), ("layers.2", "W_r1_h2_2"),
        ("head", "W_k_a__k_b"), ("layer", "Saved")]
    assert table[3]["kernel_ids"] == ["k-a", "k-a", "k.b"]  # the table keeps each path's own list


def test_bundle_loads_validates_and_benchmarks_without_the_harness(tmp_path):
    model = _fixture_model("repeated_layers.py")
    x = mx.random.normal((4, 16), key=mx.random.key(0))
    x_small = mx.random.normal((2, 16), key=mx.random.key(1))
    layer3 = _emit_layer(model, x, "layers.3@0", "ArtLayer3")
    # a second copy of the same layer emits the same body apart from its class
    # name: the export must write it once and point both module paths at it
    layer2 = _emit_layer(model, x, "layers.2@0", "ArtLayer2")
    assert layer2.source.replace("ArtLayer2", "ArtLayer3") == layer3.source

    patched = _fixture_model("repeated_layers.py")
    for emitted, index in ((layer3, 3), (layer2, 2)):
        install(patched, f"layers.{index}",
                _load_class(emitted)(patched.layers[index], {"k_art_add": ADD_KERNEL}))
    expected = {}
    for label, inputs in (("main", [x]), ("main@L=2", [x_small])):
        y = patched(*inputs)
        mx.eval(y)
        expected[label] = y
        assert mx.array_equal(y, model(*inputs)).item()  # the add kernel is exact

    report = Report()
    report.baseline = {"choice": "plain"}
    report.step_ms = {"main": {"before": 0.4, "after": 0.301, "baseline_at_end": 0.312,
                               "speedup": 1.037, "stability": 0.9, "win_confirmed": True,
                               "steps_per_sample": 7}}
    report.final = {"passed": True, "sequences": {"main": {
        "steps": 7, "baseline_sequence_ms": 0.9, "candidate_sequence_ms": 0.87, "win_confirmed": False}}}
    bundle = ModelBundle(
        model_path=FIXTURES / "repeated_layers.py", baseline="plain",
        workloads={"main": [x], "main@L=2": [x_small]},
        workload_config=(Workload(inputs=(InputSpec(shape=("L", 16), dtype="float32"),), name="main"),),
        final_benchmark={"steps": 3, "pairs": 4, "warmup_steps": 1},
    )
    cases = [(label, inputs, [expected[label]]) for label, inputs in (("main", [x]), ("main@L=2", [x_small]))]
    out = emit_artifact(tmp_path / "artifact", [ADD_KERNEL], [layer3, layer2], report,
                        validate=lambda staged: check_apply_many(staged, FIXTURES / "repeated_layers.py", cases),
                        bundle=bundle)

    # the layout
    for name in ("README.md", "bundle.json", "load.py", "apply.py", "validate.py", "benchmark.py",
                 "requirements.txt", "swap_table.json", "report.json", "__init__.py",
                 "model/tests/fixtures/repeated_layers.py", "workloads/main.safetensors",
                 "workloads/main_L_2.safetensors", "kernels/k_art_add.metal", "patch/wrappers.py",
                 "runtime/autotuner_runtime/sequence.py"):
        assert (out / name).exists(), name
    assert (out / "model" / "tests" / "fixtures" / "repeated_layers.py").read_text() == \
        (FIXTURES / "repeated_layers.py").read_text()
    meta = json.loads((out / "bundle.json").read_text())
    assert meta["entry"] == "tests/fixtures/repeated_layers.py"
    assert meta["baseline"] == "plain" and meta["mlx"] == mx.__version__
    assert meta["declared_workloads"] == {"main": [{"shape": ["L", 16], "dtype": "float32"}]}
    assert meta["workloads"]["main"] == {"file": "workloads/main.safetensors", "role": "declared",
                                         "shapes": [[4, 16]], "dtypes": ["float32"]}
    assert meta["workloads"]["main@L=2"]["role"] == "sweep"
    assert meta["final_benchmark"] == {"steps": 3, "pairs": 4, "warmup_steps": 1,
                                       "steps_by_workload": {"main": 7}, "warmup_steps_per_sample": {"main": 7}}
    assert meta["patches"] == [{"module_path": "layers.3", "kernel_ids": ["k_art_add"]},
                               {"module_path": "layers.2", "kernel_ids": ["k_art_add"]}]
    saved = mx.load(str(out / "workloads/main.safetensors"))
    assert mx.array_equal(saved["i0"], x).item()
    assert f"mlx=={mx.__version__}" in (out / "requirements.txt").read_text()

    # one class per distinct body; every path maps to it
    wrappers = (out / "patch" / "wrappers.py").read_text()
    assert wrappers.count("\nclass ") == 1 and "class W_k_art_add(ReplayWrapper):" in wrappers
    assert "# serves layers.3, layers.2\n" in wrappers
    table = json.loads((out / "swap_table.json").read_text())
    assert [(r["scope_path"], r["wrapper_class"]) for r in table] == [("layers.3", "W_k_art_add"),
                                                                       ("layers.2", "W_k_art_add")]

    readme = (out / "README.md").read_text()
    for text in ("k_art_add", "`layers.3`", "`layers.2`", "0.312 ms untouched", "0.301 ms patched",
                 "7 consecutive steps", "from artifact import load", "python validate.py",
                 "python benchmark.py", "plain model exactly as build() returns it"):
        assert text in readme, text
    assert "—" not in readme

    # a fresh process, a foreign working directory, no harness: load() and validate()
    child = textwrap.dedent(f"""
        import importlib.util, sys
        from pathlib import Path
        import mlx.core as mx
        art = Path({str(out)!r})
        def module(name):
            spec = importlib.util.spec_from_file_location(name, art / f"{{name}}.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        loader, validator = module("load"), module("validate")
        patched, original = loader.load(), loader.load(patched=False)
        assert not patched.compiled and patched.definition.__file__.startswith(str(art / "model"))
        assert "autotuner" not in sys.modules, "the bundle must not import the harness"
        for label, inputs, role in validator.saved_workloads(loader.json.loads((art / "bundle.json").read_text())):
            y = patched(*inputs)
            mx.eval(y)
            print("SUM", label, float(y.astype(mx.float32).sum()))
        results = validator.validate(original, patched)
        assert [r["passed"] for r in results] == [True, True], results
        assert [r["role"] for r in results] == ["declared", "sweep"]
        print("VALIDATED", len(results))
    """)
    proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=180,
                          cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    sums = {l.split(" ")[1]: float(l.split(" ")[2]) for l in lines if l.startswith("SUM")}
    for label, y in expected.items():
        assert abs(sums[label] - float(y.astype(mx.float32).sum())) < 1e-3
    assert "VALIDATED 2" in lines

    # validate.py and benchmark.py as scripts
    proc = subprocess.run([sys.executable, str(out / "validate.py")], capture_output=True, text=True,
                          timeout=180, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("PASS ") == 2 and "2 of 2 workloads match" in proc.stdout

    summary = tmp_path / "bench.json"
    proc = subprocess.run([sys.executable, str(out / "benchmark.py"), "--pairs", "4",
                           "--json", str(summary)], capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert proc.returncode in (0, 1), proc.stderr
    stdout = proc.stdout
    assert stdout.count("PASS ") == 3  # both workloads and the consecutive trajectory checked first
    assert "main: 7 consecutive steps, original " in stdout and "(patched over original)" in stdout
    # on a toy model the verdict is noise; the exit code must follow whatever the verdict was
    confirmed = "benchmark: confirmed speedup on at least one workload; no detected regressions" in stdout
    assert confirmed or "benchmark: no confirmed speedup" in stdout
    assert proc.returncode == (0 if confirmed else 1)
    rows = json.loads(summary.read_text())
    assert rows["steps"] == 7 and rows["pairs"] == 4 and list(rows["workloads"]) == ["main"]
    row = rows["workloads"]["main"]
    assert row["steps"] == 7 and len(row["observations"]) == 8 and row["win_confirmed"] is confirmed
    assert row["timing"]["n"] == 4 and row["baseline_sequence_ms"] > 0


def test_bundle_refuses_a_model_that_imports_the_harness(tmp_path):
    import pytest

    bad = tmp_path / "bad_model.py"
    bad.write_text("import autotuner\n\ndef build():\n    return lambda x: x\n")
    (tmp_path / "pyproject.toml").write_text("")
    with pytest.raises(ValueError, match="must not import the autotuner harness"):
        emit_artifact(tmp_path / "artifact", [], [], Report(),
                      bundle=ModelBundle(model_path=bad, baseline="plain"))
    assert not (tmp_path / "artifact").exists()


def test_bundle_changing_rule_matches_job_rule():
    from autotuner.artifact.validate import check_outputs
    from autotuner.e2e import changing_check

    original = lambda: mx.array([1.01, -1.0], dtype=mx.float32)
    tolerance = (0.0, 0.003)
    for value in ([1.008, -1.0], [1.02, -1.0], [float('nan'), -1.0]):
        candidate = lambda: mx.array(value, dtype=mx.float32)
        job = changing_check(original, candidate, 'main', tolerances=tolerance)
        exported = check_outputs(original, candidate, 'main', exact=False, tolerances=tolerance)
        assert exported['passed'] == job.passed
        assert exported['max_abs'] == job.max_abs
        assert exported['allowance'] == job.allowance
    candidate = lambda: mx.array([1.008, -1.0], dtype=mx.float32)
    assert check_outputs(original, candidate, 'main', exact=False, tolerances=tolerance)['passed']
    assert not check_outputs(original, candidate, 'main')['passed']


def test_bundle_saves_complete_golden_set(tmp_path):
    from autotuner.artifact.bundle import write_bundle
    import pytest

    x = mx.array([1.0])
    bundle = ModelBundle(model_path=FIXTURES / 'repeated_layers.py', baseline='plain',
                        workloads={'main': [x], 'sweep': [x]}, goldens={'main': [x]})
    out = tmp_path / 'bundle'
    out.mkdir()
    with pytest.raises(ValueError, match='every saved workload'):
        write_bundle(out, bundle, [], {})
    bundle.goldens['sweep'] = [x]
    metadata = write_bundle(out, bundle, [], {})
    assert set(metadata['goldens']) == {'main', 'sweep'}
    assert mx.array_equal(mx.load(str(out / metadata['goldens']['main']))['o0'], x).item()


def test_exported_changing_gate_checks_each_value_near_zero():
    from autotuner.e2e import changing_check
    from autotuner.artifact.validate import check_outputs
    original = lambda: mx.array([1e-5, 100.1])
    bad = lambda: mx.array([1.0, 100.0])
    tolerance = (0.01, 0.001)
    expected = changing_check(original, bad, 'near_zero', tolerances=tolerance)
    actual = check_outputs(original, bad, 'near_zero', exact=False, tolerances=tolerance)
    assert not expected.passed and not actual['passed']
    assert actual['max_abs'] == expected.max_abs


def test_export_cleans_early_write_failure_and_preserves_previous(tmp_path, monkeypatch):
    import pytest
    import importlib
    emitter = importlib.import_module('autotuner.artifact.emit')
    target = tmp_path / 'artifact'
    target.mkdir()
    (target / 'keep').write_text('previous accepted artifact')
    def broken_write(*args):
        raise OSError('simulated disk full')
    monkeypatch.setattr(emitter, 'write_kernel', broken_write)
    with pytest.raises(OSError, match='disk full'):
        emit_artifact(target, [ADD_KERNEL], [], Report())
    assert (target / 'keep').read_text() == 'previous accepted artifact'
    assert not list(tmp_path.glob('artifact.building-*'))


def test_export_rolls_back_failed_publish(tmp_path, monkeypatch):
    import pytest
    target = tmp_path / 'artifact'
    target.mkdir()
    (target / 'keep').write_text('previous accepted artifact')
    rename = Path.rename
    def fail_publish(path, destination):
        if '.building-' in path.name:
            raise OSError('simulated publication failure')
        return rename(path, destination)
    monkeypatch.setattr(Path, 'rename', fail_publish)
    with pytest.raises(OSError, match='publication failure'):
        emit_artifact(target, [], [], Report())
    assert (target / 'keep').read_text() == 'previous accepted artifact'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['artifact']


def test_changing_decode_bundle_requires_sequence_reference(tmp_path):
    import pytest
    from autotuner.artifact.bundle import write_bundle
    x = mx.array([[1]])
    bundle = ModelBundle(FIXTURES / 'repeated_layers.py', workloads={'decode':[x]},
                        goldens={'decode':[x]}, context={'workload':'decode'})
    with pytest.raises(ValueError, match='consecutive-step fp32 reference'):
        write_bundle(tmp_path, bundle, [], {})


def test_unpackageable_dependency_fails_before_model_build(tmp_path):
    import pytest
    from autotuner.artifact.bundle import check_model_dependencies
    model = tmp_path / 'model.py'
    model.write_text('import missing_autotune_test_dependency\ndef build(): pass\n')
    with pytest.raises(ValueError, match='missing_autotune_test_dependency'):
        check_model_dependencies(model)


def test_dynamic_checkpoint_passes_source_dependency_preflight(tmp_path):
    import pytest
    from autotuner.artifact.bundle import check_model_dependencies
    model = tmp_path / 'model.py'
    model.write_text('import os\nfrom mlx_lm import load\ndef build():\n return load(os.environ["MODEL"])[0]\n')
    check_model_dependencies(model)


def test_partial_recovery_goldens_cannot_silently_use_preserving_rule(tmp_path, monkeypatch):
    import pytest
    from autotuner.artifact import validate as validator
    (tmp_path / 'bundle.json').write_text(json.dumps({
        'correctness_rule':'fp32_relative', 'workloads':{'main':{},'sweep':{}},
        'goldens':{'main':'main.golden.safetensors'}}))
    monkeypatch.setattr(validator, '_HERE', tmp_path)
    with pytest.raises(ValueError, match='lacks fp32 references for: sweep'):
        validator.validate()


def test_bundle_records_tolerance_without_fp32_references(tmp_path):
    from autotuner.artifact.bundle import write_bundle
    x = mx.array([1.0])
    bundle = ModelBundle(model_path=FIXTURES / 'repeated_layers.py', baseline='plain',
                        workloads={'main': [x]}, exact=False, tolerances=(0.002, 0.0001))
    metadata = write_bundle(tmp_path, bundle, [], {})
    assert metadata['correctness_rule'] == 'baseline_tolerance'
    assert metadata['tolerances'] == {'rtol': 0.002, 'atol': 0.0001}
    assert metadata['goldens'] == {} and metadata['sequence_goldens'] == {}
    assert 'float32' in metadata['tolerance_defaults']


def test_original_instability_does_not_widen_tolerance():
    from autotuner.artifact.validate import check_outputs
    original = iter((mx.array([1.0]), mx.array([1.1])))
    result = check_outputs(lambda: next(original), lambda: mx.array([1.05]),
                           'unstable', exact=False, tolerances=(0.0, 0.01))
    assert not result['passed'] and 'not bitwise repeatable' in result['reason']


def test_export_tolerance_uses_untouched_model_for_cumulative_error():
    from autotuner.artifact.validate import check_outputs
    # Each of two edits adds .006, but the complete model has one .01 allowance.
    original = lambda: mx.array([1.0])
    first = lambda: mx.array([1.006])
    combined = lambda: mx.array([1.012])
    assert check_outputs(original, first, 'first', exact=False, tolerances=(0, .01))['passed']
    assert not check_outputs(original, combined, 'combined', exact=False, tolerances=(0, .01))['passed']


def test_export_exact_preserves_signed_zero_and_nonfinite_payloads():
    from autotuner.artifact.validate import check_outputs
    original = lambda: mx.array([0.0, float('inf'), float('nan')])
    changed = lambda: mx.array([-0.0, float('inf'), float('nan')])
    assert not check_outputs(original, changed, 'zero')['passed']
    assert check_outputs(original, changed, 'zero', exact=False, tolerances=(0, 0))['passed']


def test_bundle_tolerance_validation_runs_fresh_without_optimizer(tmp_path):
    model = tmp_path / 'model.py'
    model.write_text('def build():\n    return lambda x: x\n')
    bundle = ModelBundle(model_path=model, project_root=tmp_path, baseline='plain',
                        workloads={'main': [mx.array([1.0])]}, exact=False,
                        tolerances=(0.0, 0.01))
    out = emit_artifact(tmp_path / 'artifact', [], [], Report(), bundle=bundle)
    child = textwrap.dedent(f"""
        import importlib.util, sys
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        spec = importlib.util.spec_from_file_location('validator', {str(out / 'validate.py')!r})
        validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator)
        good = validator.validate(lambda x: x, lambda x: x + 0.005)
        bad = validator.validate(lambda x: x, lambda x: x + 0.02)
        assert good[0]['passed'] and not bad[0]['passed'], (good, bad)
        assert 'autotuner' not in sys.modules
    """)
    proc = subprocess.run([sys.executable, '-c', child], cwd=tmp_path,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_tolerance_never_allows_nondeterministic_model_outputs():
    from autotuner.artifact.validate import check_outputs
    original = lambda: mx.array([1.0])
    values = iter((mx.array([1.001]), mx.array([1.002])))
    result = check_outputs(original, lambda: next(values), 'candidate_repeat',
                           exact=False, tolerances=(0, .01))
    assert not result['passed'] and 'patched model is not bitwise repeatable' in result['reason']
    values = iter((mx.array([1.001]), mx.array([1.002])))
    result = check_outputs(lambda: next(values), original, 'original_repeat',
                           exact=False, tolerances=(0, .01))
    assert not result['passed'] and 'original model is not bitwise repeatable' in result['reason']


def test_output_structure_is_frozen_before_reused_container_mutation():
    from autotuner.artifact.validate import check_outputs
    original = lambda: {'out': mx.array([1.0]), 'counter': 1}
    shared = {'out': mx.array([1.0]), 'counter': 0}
    values = iter((2, 1))
    def candidate():
        shared['counter'] = next(values)
        return shared
    result = check_outputs(original, candidate, 'shared_dict')
    assert not result['passed'] and 'structure' in result['reason']


def test_random_weight_bundle_preserves_build_and_validates_shared_weights(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    model_path = source / 'model.py'
    model_path.write_text((FIXTURES / 'repeated_layers.py').read_text().replace('        mx.random.seed(20)\n', ''))
    spec = importlib.util.spec_from_file_location('random_model', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build()
    x = mx.ones((4, 16))
    emitted = _emit_layer(model, x, 'layers.3@0', 'RandomLayer')
    install(model, 'layers.3', _load_class(emitted)(model.layers[3], {ADD_KERNEL.kernel_id: ADD_KERNEL}))
    expected = model(x)
    mx.eval(expected)
    bundle = ModelBundle(model_path, baseline='plain', project_root=source,
                         workloads={'main': [x]}, workload_config={'main': []},
                         final_benchmark={'steps': 2, 'pairs': 4, 'warmup_steps': 1})
    out = emit_artifact(tmp_path / 'artifact', [ADD_KERNEL], [emitted], Report(), bundle=bundle,
                        validate=lambda staged: check_apply_many(staged, model_path, [('main', [x], [expected])]))
    assert (out / 'model/model.py').read_bytes() == model_path.read_bytes()
    assert not list((out / 'model').rglob('*.safetensors'))
    code = '''
import importlib.util, sys
import mlx.core as mx
spec = importlib.util.spec_from_file_location('loader', sys.argv[1] + '/load.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a = m.load(patched=False)
b = m.load(patched=False)
assert not mx.array_equal(a.model.layers[0].w, b.model.layers[0].w).item()
p = m.load(share_weights_with=a.model)
from autotuner_runtime import kernels
from autotuner_runtime.exact import bitwise_equal
calls = []
original_call = kernels.call
def counted(*args, **kwargs):
    calls.append(True)
    return original_call(*args, **kwargs)
kernels.call = counted
x = mx.ones((4,16))
assert bitwise_equal(a(x), p(x))
assert calls, 'exported kernel never ran'
assert 'autotuner' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(out)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_random_cached_bundle_shares_weights_before_filling_context(tmp_path):
    source = tmp_path / 'model.py'
    text = (FIXTURES / 'context_model.py').read_text()
    text = text.replace('mx.arange(VOCAB * WIDTH, dtype=mx.float32).reshape(VOCAB, WIDTH) / 64',
                        'mx.random.normal((VOCAB, WIDTH))')
    source.write_text(text)
    x = mx.array([[1, 2]], dtype=mx.int32)
    prefix = mx.array([[3, 4, 5]], dtype=mx.int32)
    bundle = ModelBundle(source, baseline='plain', project_root=tmp_path,
        workloads={'prompt': [x]}, workload_config={'prompt': []},
        context={'workload': 'prompt', 'context': 3, 'seed': 0, 'tokens': prefix},
        final_benchmark={'steps': 2, 'pairs': 4, 'warmup_steps': 1})
    out = emit_artifact(tmp_path / 'artifact', [], [], Report(), bundle=bundle,
        validate=lambda staged: check_apply_many(staged, source, [('prompt', [x], [])],
                                                 context=(3, prefix, [x])))
    assert (out / 'model/model.py').read_bytes() == source.read_bytes()


def test_run_weight_sharing_copies_the_matching_prefix_state():
    from autotuner.e2e import share_weights
    from autotuner_runtime.state import context_step, _cache_observation
    from autotuner.artifact.validate import check_outputs
    from tests.fixtures.context_model import build
    first, second = build(), build()
    second.embed.weight = second.embed.weight + 3.0
    prefix = mx.array([[1, 2, 3]], dtype=mx.int32)
    x = mx.array([[4]], dtype=mx.int32)
    a = context_step(first, 3, prefix, [x])
    b = context_step(second, 3, prefix, [x])
    assert not check_outputs(lambda: a(x), lambda: b(x), 'different weights')['passed']
    share_weights(a, b)
    assert a._cache[0] is not b._cache[0]
    assert check_outputs(lambda: a.correctness(x), lambda: b.correctness(x), 'same weights')['passed']
    b._cache[0].offset += 1
    assert a._cache[0].offset == 3
