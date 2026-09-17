"""A workload with a context runs as one repeatable step over the model's
own KV cache: the harness builds, fills and rewinds the cache, so the model
file is the plain model. The fixture follows mlx-lm's cache conventions."""

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner.bind.verify import verify_retrace
from autotuner.ladder.gates import LadderResult
from autotuner.loop import _resolve, JobRunner, RegionRun, _state_marks
from autotuner.manifest import ManifestError
from autotuner.measure.session import Session
from autotuner_runtime.kernels import KernelSpec
from autotuner_runtime.state import ContextStep, context_step

FIXTURES = Path(__file__).parent / "fixtures"
MODEL = FIXTURES / "context_model.py"


def _fixture():
    spec = importlib.util.spec_from_file_location("ctx_fixture", MODEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest(tmp_path, context, model=MODEL, name="decode", tolerances=None):
    line = "" if context is None else f"\n    context: {context}"
    manifest = tmp_path / f"manifest_{name}_{context}.yaml"
    manifest.write_text(f"""model: {model}
baseline: plain
workloads:
  - name: {name}
    inputs: [{{shape: [1, 1], dtype: int32, high: 32}}]{line}
""")
    if tolerances is not None:
        with manifest.open("a") as stream:
            stream.write(f"tolerances: {{rtol: {tolerances[0]}, atol: {tolerances[1]}}}\n")
    return manifest


def _runner(tmp_path, context, model=MODEL, monkeypatch=None, tolerances=None):
    if monkeypatch is not None:
        # only the speed decision is replaced: identity, correctness, the live
        # dispatch, and the literal retrace all still have to pass
        monkeypatch.setattr(JobRunner, "_model_win", lambda self, result: all(c.passed for c in result.checks))
    runner = JobRunner(_manifest(tmp_path, context, model, tolerances=tolerances), tmp_path / f"work_{context}",
                       judge_factory=lambda region: None, clock_pairs=4,
                       session=Session(sleep=lambda seconds: None))
    runner.load_model()
    return runner


def _outputs(runner, n=3):
    outputs = [runner.model(*runner.tensors["decode"]) for _ in range(n)]
    mx.eval(outputs)
    return outputs


def test_the_step_is_the_same_call_every_time(tmp_path):
    runner = _runner(tmp_path, context=4)
    try:
        runner.trace_workloads()
        fx = _fixture()
        assert isinstance(runner.model, ContextStep) and isinstance(runner.baseline_model, ContextStep)
        assert runner.context_tokens.shape == (1, 4)
        caches = runner.model._cache
        # Restore the entire prefix, including allocation state, after calls.
        assert [c.offset for c in caches] == [4, 4]
        assert caches[0].keys.shape == (1, fx.STEP, fx.WIDTH)
        outputs = _outputs(runner, 5)
        assert [c.offset for c in caches] == [4, 4]
        assert caches[0].keys.shape == (1, fx.STEP, fx.WIDTH)
        assert all(mx.array_equal(outputs[0], o).item() for o in outputs[1:])
        # the baseline copy is the same step over the same context
        assert mx.array_equal(runner.baseline_model(*runner.tensors["decode"]), outputs[0]).item()
        assert all(mx.array_equal(a.keys, b.keys).item()
                   for a, b in zip(caches, runner.baseline_model._cache))
        # the trace: one state call per layer, so the job takes the plain baseline
        trace = runner.traces["decode"]
        assert [n.op for n in trace.nodes if n.op.startswith("state:")] == (
            ["state:Cache.update_and_fetch"] * 2 + ["state:Cache.__getattribute__"] * 2)
        assert _state_marks(trace).endswith("4 state calls")  # two updates, two completion reads
        assert runner.log.rows()[0]["context"] == {"workload": "decode", "tokens": 4, "seed": runner.context_seed}
    finally:
        runner.tracer.uninstall()


def test_the_context_changes_the_step(tmp_path):
    """The number is used: each layer sums every stored key, so a longer
    context gives a different output on the same token."""
    outputs, tokens = [], []
    for context in (2, 6):
        runner = _runner(tmp_path, context=context)
        try:
            runner.trace_workloads()
            tokens.append(runner.tensors["decode"][0])
            outputs.append(_outputs(runner)[0])
        finally:
            runner.tracer.uninstall()
    assert mx.array_equal(*tokens).item()
    assert not mx.array_equal(*outputs).item()


def test_no_context_is_the_plain_model(tmp_path):
    runner = _runner(tmp_path, context=None)
    try:
        assert not isinstance(runner.model, ContextStep) and runner.context is None
        runner.trace_workloads()
        assert _state_marks(runner.traces["decode"]) == ""
        assert runner.log.rows()[0]["context"] is None
    finally:
        runner.tracer.uninstall()


def test_context_zero_is_an_empty_cache(tmp_path):
    runner = _runner(tmp_path, context=0)
    try:
        runner.trace_workloads()
        assert [c.offset for c in runner.model._cache] == [0, 0]
        outputs = _outputs(runner)
        assert [c.offset for c in runner.model._cache] == [0, 0]
        assert all(mx.array_equal(outputs[0], o).item() for o in outputs[1:])
    finally:
        runner.tracer.uninstall()


def test_a_model_without_make_cache_is_refused_with_the_fix(tmp_path):
    with pytest.raises(ManifestError, match="make_cache"):
        _runner(tmp_path, context=3, model=FIXTURES / "sibling_children.py")


def test_a_cache_without_trim_can_be_restored(tmp_path):
    model = tmp_path / "frozen.py"
    model.write_text(textwrap.dedent("""
        import mlx.nn as nn

        class Frozen:
            offset = 0
            def is_trimmable(self):
                return False

        class Model(nn.Module):
            def __call__(self, tokens, cache=None):
                return tokens
            def make_cache(self):
                return [Frozen()]

        def build():
            return Model()
    """))
    runner = _runner(tmp_path, context=3, model=model)
    try:
        runner.trace_workloads()
        assert mx.array_equal(*_outputs(runner, 2)).item()
    finally:
        runner.tracer.uninstall()


def test_a_model_that_takes_no_cache_is_refused(tmp_path):
    model = tmp_path / "nocache.py"
    model.write_text(textwrap.dedent("""
        import mlx.nn as nn

        class Model(nn.Module):
            def __call__(self, tokens):
                return tokens
            def make_cache(self):
                return []

        def build():
            return Model()
    """))
    with pytest.raises(ManifestError, match="cache"):
        _runner(tmp_path, context=3, model=model)


MUL_KERNEL = KernelSpec(
    kernel_id="ctx_mul", name="ctx_mul", input_names=("x", "w"), output_names=("out",),
    source="uint i = thread_position_in_grid.x; out[i] = x[i] * w[i % 8];",
    grid=("in0.shape[0] * in0.shape[1] * in0.shape[2]", "1", "1"), threadgroup=("8", "1", "1"),
    output_shapes=(("in0.shape[0]", "in0.shape[1]", "in0.shape[2]"),), output_dtypes=("float32",),
)


def test_a_kernel_binds_inside_the_step_and_the_bundle_rebuilds_it(tmp_path, monkeypatch):
    """The multiply feeding each layer's cache write gets a kernel: bind,
    retrace, the whole-model check, then the artifact rebuilt in a fresh
    process through apply() on the plain model file and load() on the
    bundle, each around the same filled cache."""
    runner = _runner(tmp_path, context=4, monkeypatch=monkeypatch)
    try:
        runner.trace_workloads()
        regions = runner.build_regions()
        region = next(r for r in regions if r.ops == ("array.__mul__",))
        assert {m.scope_stack[-1] for m in region.members} == {"model.layers.0@0", "model.layers.1@0"}
        assert runner._bind_and_promote(
            RegionRun(region=region), MUL_KERNEL,
            LadderResult("tentative_ship", None, {}, 1.0, 2.0, 1.0, 0.0, [])), runner.log.rows()[-1]
        assert set(runner.installed) == {"model.layers.0", "model.layers.1"}
        x = runner.tensors["decode"]
        # The layer compiles at the recorded cache position (graph delivery):
        # each layer's one call substitutes both copies of the cut, and a
        # retrace shows no library multiply where the kernel now runs.
        from collections import Counter
        from contextlib import ExitStack
        from autotuner_runtime.graph import GraphWrapper
        wrappers = [_resolve(runner.model, path) for path in ("model.layers.0", "model.layers.1")]
        assert all(isinstance(w, GraphWrapper) for w in wrappers)
        with ExitStack() as stack:
            calls = [stack.enter_context(w.validate_graph()) for w in wrappers]
            runner.model(*x)
        assert [Counter(row["kernel_id"] for call in c for row in call if row["hits"]) for c in calls] \
            == [{MUL_KERNEL.kernel_id: 2}] * 2
        runner.tracer.install()
        retrace, _ = runner.tracer.trace(runner.model, x)
        assert "array.__mul__" not in {n.op for n in retrace.nodes}
        runner.tracer.uninstall()
        assert [c.offset for c in runner.model._cache] == [4, 4]
        expected = _outputs(runner)
        assert all(mx.array_equal(expected[0], o).item() for o in expected[1:])
        assert mx.array_equal(expected[0], runner.baseline_model(*x)).item()

        runner.final_ok = True
        out = runner._emit_artifact(tmp_path / "artifact")
        meta = json.loads((out / "bundle.json").read_text())
        assert meta["context"] == {"workload": "decode", "context": 4, "seed": runner.context_seed,
                                   "file": "workloads/decode.context.safetensors"}
        saved = mx.load(str(out / meta["context"]["file"]))["tokens"]
        assert mx.array_equal(saved, runner.context_tokens).item()
        assert [p["module_path"] for p in meta["patches"]] == ["model.layers.0", "model.layers.1"]
        assert "a KV cache already holding 4 tokens" in (out / "README.md").read_text()
    finally:
        runner.tracer.uninstall()
        assert runner.tracer.verify_restored() == []

    # a fresh process, no harness: load() is the same step over the same context
    child = textwrap.dedent(f"""
        import importlib.util, sys
        from pathlib import Path
        import mlx.core as mx
        art = Path({str(out)!r})
        spec = importlib.util.spec_from_file_location("load", art / "load.py")
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        original = loader.load(patched=False)
        patched = loader.load(share_weights_with=original.model)  # as benchmark.py loads them
        assert "autotuner" not in sys.modules
        for m in (patched, original):
            assert type(m.model).__name__ == "ContextStep"
            assert [c.offset for c in m.model._cache] == [4, 4]
        x = mx.load(str(art / "workloads/decode.safetensors"))["i0"]
        ys = [patched(x) for _ in range(3)] + [original(x)]
        mx.eval(ys)
        assert all(mx.array_equal(ys[0], y).item() for y in ys[1:])
        assert [c.offset for c in patched.model._cache] == [4, 4]
        print("SUM", float(ys[0].sum()))
    """)
    proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=180,
                          cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert float(proc.stdout.split("SUM ")[1]) == pytest.approx(float(expected[0].sum()))
    proc = subprocess.run([sys.executable, str(out / "validate.py")], capture_output=True, text=True,
                          timeout=180, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "1 of 1 workloads match" in proc.stdout


def test_context_step_restores_all_position_fields():
    """A rotating cache moves two positions per token; both must reset."""
    fx = _fixture()

    class Rotating(fx.Cache):
        def __init__(self):
            super().__init__()
            self.idx = 0

        def update_and_fetch(self, keys):
            self.idx += keys.shape[1]
            return super().update_and_fetch(keys)

        def trim(self, n):
            n = super().trim(n)
            self.idx -= n
            return n

    model = fx.build()
    model.make_cache = lambda: [Rotating() for _ in model.layers]
    tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
    step = context_step(model, 3, tokens, [mx.array([[5]], dtype=mx.int32)])
    for _ in range(3):
        mx.eval(step(mx.array([[7]], dtype=mx.int32)))
    assert [(c.offset, c.idx) for c in step._cache] == [(3, 3), (3, 3)]


def test_reordering_kernel_on_stateful_model_uses_original_and_manifest_tolerance(tmp_path, monkeypatch):
    """Reordering reaches whole-model validation without an FP32 interpreter.
    Incorrect output/state is rejected; exact math tagged changing can ship."""
    runner = _runner(tmp_path, context=4, monkeypatch=monkeypatch, tolerances=(1e-5, 1e-6))
    try:
        runner.trace_workloads()
        assert [c.offset for c in runner.baseline_model._cache] == [4, 4]
        region = next(r for r in runner.build_regions() if r.ops == ("array.__mul__",))
        result = LadderResult("tentative_ship", None, {}, 1.0, 2.0, 1.0, 0.0, [])

        wrong = KernelSpec(**{**MUL_KERNEL.__dict__, "kernel_id": "ctx_mul_wrong", "name": "ctx_mul_wrong",
                              "source": "uint i = thread_position_in_grid.x; out[i] = x[i] * w[i % 8] * 1.01f;"})
        assert not runner._bind_and_promote(RegionRun(region=region), wrong, result, assoc_tag="changing")
        assert not runner.installed
        assert runner.log.rows()[-1]["kind"] == "not_shipped"
        assert runner.log.rows()[-1]["reason"] == "outputs changed"

        assert runner._bind_and_promote(RegionRun(region=region), MUL_KERNEL, result, assoc_tag="changing")
        assert set(runner.installed) == {"model.layers.0", "model.layers.1"}
        assert not runner._requires_exact()
        assert runner.manifest.tolerances == (1e-5, 1e-6)
        assert not any(row["kind"] == "model_golden" for row in runner.log.rows())
    finally:
        runner.tracer.uninstall()
        assert runner.tracer.verify_restored() == []


def test_failed_step_rewinds_layers_that_already_advanced():
    class Cache:
        def __init__(self):
            self.offset = 4

        def trim(self, n):
            self.offset -= n

    class BrokenModel:
        def __call__(self, *inputs, cache):
            cache[0].offset += 1
            raise RuntimeError("failed in the second layer")

    caches = [Cache(), Cache()]
    step = ContextStep(BrokenModel(), caches, 4)
    with pytest.raises(RuntimeError, match="second layer"):
        step()
    assert [c.offset for c in caches] == [4, 4]


def test_sequence_advances_and_restarts_from_the_same_prefix():
    fx = _fixture()
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    tok = mx.array([[5]], dtype=mx.int32)
    model = fx.build()
    step = context_step(model, 4, tokens, [tok])
    handles = [c.keys for c in step._cache]
    before = [mx.array(c.keys) for c in step._cache]
    mx.eval(before)
    first = step.sequence([tok], 3)
    second = step.sequence([tok], 3)

    # Independent advancing run, without ContextStep.
    expected_cache = model.make_cache()
    mx.eval(model(tokens, cache=expected_cache))
    expected = []
    for _ in range(3):
        expected.append(model(tok, cache=expected_cache))
        mx.eval(expected[-1])
    assert [c.offset for c in expected_cache] == [7, 7]
    assert all(mx.array_equal(a, b).item() for a, b in zip(first, expected))
    assert all(mx.array_equal(a, b).item() for a, b in zip(first, second))
    assert not mx.array_equal(first[0], first[-1]).item()
    assert [c.offset for c in step._cache] == [4, 4]
    assert all(c.keys is a for c, a in zip(step._cache, handles))
    assert all(mx.array_equal(c.keys, a).item() for c, a in zip(step._cache, before))


def test_failed_sequence_preserves_the_live_prefix():
    class Cache:
        def __init__(self):
            self.offset = 4
            self.keys = mx.array([2.0])

    class BrokenModel:
        def __call__(self, *inputs, cache):
            cache[0].keys[0] = 9.0
            cache[0].offset += 1
            mx.eval(cache[0].keys)
            raise RuntimeError("sequence failed")

    cache = Cache()
    step = ContextStep(BrokenModel(), [cache], 4)
    original = cache.keys
    with pytest.raises(RuntimeError, match="sequence failed"):
        step.sequence([], 3)
    assert cache.offset == 4 and cache.keys is original
    assert cache.keys.item() == 2.0


def test_final_decode_sequence_checks_tolerant_trajectory_and_state(tmp_path, monkeypatch):
    from autotuner.e2e import WorkloadCheck
    from dataclasses import asdict
    from autotuner.measure.clocks import comparison_from_samples
    runner = _runner(tmp_path, 4, tolerances=(1e-5, 1e-6))
    runner.trace_workloads()
    runner.tracer.uninstall()
    runner.shipped_tags['test'] = 'changing'
    def measured(session, a, b, *warm, **kwargs):
        assert len(a()) == len(b()) == runner.manifest.final_benchmark.steps
        return {'timing': asdict(comparison_from_samples([100.] * 4, [90.] * 4)),
                'win_confirmed': True, 'speedup': 100 / 90}
    from autotuner_runtime.state import sequence_observation
    observations = []
    def observe(*args):
        value = sequence_observation(*args)
        observations.append(value)
        return value
    monkeypatch.setattr('autotuner_runtime.state.sequence_observation', observe)
    monkeypatch.setattr('autotuner.loop.compare_sequences', measured)
    rows, result = runner._final_sequences([WorkloadCheck('decode', 0, 0, 0, 1, True)])
    assert result.passed
    assert rows['decode']['workload_kind'] == 'advancing_cache_fixed_tokens'
    assert result.checks[-1].name == 'decode:sequence'
    assert result.checks[-1].rule == 'baseline_tolerance'
    assert len(observations) == 4  # both models repeat independently
    for observation in observations:
        assert len(observation['outputs']) == runner.manifest.final_benchmark.steps
        assert [cache['offset'] for cache in observation['state']] == [
            4 + runner.manifest.final_benchmark.steps] * 2
    assert [c.offset for c in runner.model._cache] == [4, 4]


def test_exported_decode_benchmark_checks_the_advancing_trajectory(tmp_path, monkeypatch):
    from autotuner.e2e import WorkloadCheck
    from autotuner.measure.clocks import comparison_from_samples
    from dataclasses import asdict
    runner = _runner(tmp_path, 4, tolerances=(1e-5, 1e-6))
    runner.trace_workloads()
    runner.tracer.uninstall()
    runner.shipped_tags['fixture'] = 'changing'
    monkeypatch.setattr('autotuner.loop.compare_sequences', lambda *a, **kw: {
        'timing': asdict(comparison_from_samples([100.] * 4, [90.] * 4)),
        'win_confirmed': True, 'speedup': 100 / 90})
    runner._final_sequences([WorkloadCheck('decode', 0, 0, 0, 1, True)])
    runner.final_ok = True
    artifact = runner.emit_artifact(tmp_path / 'artifact')
    metadata = json.loads((artifact / 'bundle.json').read_text())
    assert metadata['correctness_rule'] == 'baseline_tolerance'
    assert metadata['tolerances'] == {'rtol': 1e-5, 'atol': 1e-6}
    assert metadata['goldens'] == {} and metadata['sequence_goldens'] == {}
    result = subprocess.run([sys.executable, str(artifact / 'benchmark.py'), '--json', str(tmp_path / 'result.json')],
                            capture_output=True, text=True, timeout=45, cwd=tmp_path)
    assert result.returncode in (0, 1), result.stderr
    assert 'PASS decode:sequence' in result.stdout, result.stdout + result.stderr
    report = json.loads((tmp_path / 'result.json').read_text())
    assert report['workloads']['decode']['workload_kind'] == 'advancing_cache_fixed_tokens'
