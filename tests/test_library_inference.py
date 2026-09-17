"""The native library task is the same task in tracing, timing and correctness."""

from types import SimpleNamespace
import subprocess
import sys
import textwrap

import mlx.core as mx
import pytest

from autotuner.manifest import InputSpec, Workload
from autotuner_runtime.inference import LibraryInference, resolve_library_inference
from autotuner_runtime.state import _cache_observation, correctness_call, sequence_observation


def _build(kind):
    mx.random.seed(19)
    if kind == "llama":
        from tests.fixtures.llama_cache_model import build
        return build()
    if kind == "qwen":
        from tests.fixtures.qwen_cache_model import build
        from autotuner_runtime.captured_kernels import capture_construction
        # These definitions are imported once per process. Keep later tracer
        # tests valid even when this numerical test constructs the model first.
        with capture_construction():
            return build()
    if kind == "mamba":
        from mlx_lm.models.mamba import Model, ModelArgs
        return Model(ModelArgs(model_type="mamba", vocab_size=64, hidden_size=32,
                               intermediate_size=64, state_size=8, num_hidden_layers=2,
                               conv_kernel=4, use_bias=False, use_conv_bias=True,
                               time_step_rank=4))
    from mlx_lm.models.recurrent_gemma import Model, ModelArgs
    return Model(ModelArgs(model_type="recurrent_gemma", attention_bias=False,
                           conv1d_width=4, hidden_size=64, intermediate_size=128,
                           logits_soft_cap=30., num_attention_heads=4,
                           num_hidden_layers=2, num_key_value_heads=1,
                           rms_norm_eps=1e-6, rope_theta=10000.,
                           attention_window_size=8, vocab_size=64,
                           block_types=["recurrent", "attention"]))


def _assert_equal(a, b):
    if isinstance(a, mx.array):
        assert isinstance(b, mx.array) and a.shape == b.shape and a.dtype == b.dtype
        assert mx.array_equal(a, b).item()
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_equal(x, y)
    else:
        assert a == b


def _native(model, tokens, steps, prefix=None):
    from mlx_lm.generate import generate_step, generation_stream
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    if prefix is not None:
        list(generate_step(prefix[0], model, max_tokens=0, prompt_cache=cache))
        mx.eval([item.state for item in cache])
    generated = list(generate_step(tokens[0], model, max_tokens=steps, prompt_cache=cache))
    mx.eval([item[1] for item in generated], [item.state for item in cache])
    mx.synchronize(generation_stream)
    return {"tokens": tuple(item[0] for item in generated),
            "logprobs": tuple(item[1] for item in generated),
            "state": _cache_observation(cache)}


@pytest.mark.parametrize("kind", ["llama", "qwen", "mamba", "recurrent_gemma"])
@pytest.mark.parametrize("context,steps", [(0, 1), (0, 3), (5, 3)])
def test_runtime_matches_unmodified_library_with_independent_state(kind, context, steps):
    model = _build(kind)
    tokens = mx.array([[1, 2, 3, 4]], mx.int32)
    prefix = mx.array([[7, 8, 9, 10, 11]], mx.int32) if context else None
    runtime = LibraryInference(model, steps=steps, prefix_tokens=prefix)
    expected = _native(model, tokens, steps, prefix)
    for _ in range(2):
        _assert_equal(runtime.correctness(tokens), expected)
    result = runtime(tokens)
    mx.eval(result)
    assert set(result) == {"tokens", "logprobs"}
    _assert_equal(result, {key: expected[key] for key in result})
    _assert_equal(runtime(tokens[0]), result)


@pytest.mark.parametrize("kind", ["llama", "qwen", "mamba", "recurrent_gemma"])
def test_actual_inference_trace_can_capture_the_same_boundaries(kind):
    # Custom definitions must be captured before their model module first imports,
    # exactly like production startup. Earlier tests have already imported them.
    script = textwrap.dedent("""
        import sys
        import mlx.core as mx
        from autotuner.regions.price import capture_boundaries
        from autotuner.trace import Tracer
        from autotuner_runtime.inference import LibraryInference
        from tests.test_library_inference import _build
        tracer = Tracer()
        tracer.install()
        try:
            runtime = LibraryInference(_build(sys.argv[1]), steps=2)
            tokens = mx.array([[1, 2, 3, 4]], mx.int32)
            trace, output = tracer.trace(runtime, [tokens])
            assert len(output['tokens']) == 2
            assert any('matmul' in node.op for node in trace.nodes)
            captured = capture_boundaries(tracer, runtime, [tokens], trace, set(trace.step_outputs))
            assert len(captured) == len(trace.step_outputs) == 2
            for actual in captured.values():
                assert any(mx.array_equal(actual, value).item() for value in output['logprobs'])
        finally:
            tracer.uninstall()
    """)
    result = subprocess.run([sys.executable, "-c", script, kind], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_correctness_and_sequence_observation_do_not_square_generation_length():
    runtime = LibraryInference(_build("llama"), steps=3)
    tokens = [mx.array([[1, 2, 3]], mx.int32)]
    for model in (runtime, SimpleNamespace(model=runtime)):
        assert len(correctness_call(model, tokens)["tokens"]) == 3
        assert len(sequence_observation(model, tokens, 3)["tokens"]) == 3


def test_support_uses_model_contract_and_shapes_not_workload_labels():
    model = _build("llama")
    for name in ("prefill", "decode", "denoise", "banana"):
        workload = Workload((InputSpec((1, "T"), "int32"),), name)
        assert resolve_library_inference(model, None, [workload], dims={"T": 4})
        assert resolve_library_inference(model, True, [workload], dims={"T": 4})
        assert not resolve_library_inference(model, False, [workload], dims={"T": 4})


@pytest.mark.parametrize("inputs", [
    (InputSpec((2, 4), "int32"),),
    (InputSpec((1, 4), "float16"),),
    (InputSpec((1, 4), "int32"), InputSpec((1, 4), "int32")),
])
def test_unsupported_shapes_fail_explicit_mode_and_default_to_forward(inputs):
    model = _build("llama")
    workload = Workload(inputs, "label")
    assert not resolve_library_inference(model, None, [workload])
    with pytest.raises(ValueError, match="unsupported"):
        resolve_library_inference(model, True, [workload])


def test_arbitrary_model_does_not_accidentally_become_language_generation():
    workload = Workload((InputSpec((1, 4), "int32"),), "decode")
    model = lambda tokens, cache=None: tokens
    assert not resolve_library_inference(model, None, [workload])
    with pytest.raises(ValueError, match="complete MLX-LM"):
        resolve_library_inference(model, True, [workload])


def test_transformer_body_without_vocabulary_head_is_not_an_inference_model():
    workload = Workload((InputSpec((1, 4), "int32"),), "label")
    bare_body = _build("llama").model
    assert not resolve_library_inference(bare_body, None, [workload])
    with pytest.raises(ValueError, match="complete MLX-LM"):
        resolve_library_inference(bare_body, True, [workload])


def test_batch_sweeps_are_validated_before_selecting_library_inference():
    workload = Workload((InputSpec(("B", "T"), "int32"),), "label")
    model = _build("llama")
    kwargs = {"dims": {"B": 1, "T": 4}, "sweep": {"B": [1, 2], "T": [1, 4]}}
    assert not resolve_library_inference(model, None, [workload], **kwargs)
    with pytest.raises(ValueError, match="correctness sweep"):
        resolve_library_inference(model, True, [workload], **kwargs)


def test_refresh_prefix_uses_replaced_weights():
    model = _build("llama")
    tokens = mx.array([[1, 2, 3]], mx.int32)
    prefix = mx.array([[7, 8, 9]], mx.int32)
    runtime = LibraryInference(model, steps=2, prefix_tokens=prefix)
    before = runtime.correctness(tokens)
    model.update(model.apply(lambda a: mx.zeros_like(a)).parameters())
    runtime.refresh_prefix()
    _assert_equal(runtime.correctness(tokens), _native(model, tokens, 2, prefix))
    assert not mx.array_equal(before["logprobs"][0], runtime(tokens)["logprobs"][0]).item()


def test_interrupted_generation_does_not_change_the_next_trials_prefix(monkeypatch):
    import importlib
    generation = importlib.import_module("mlx_lm.generate")
    runtime = LibraryInference(_build("llama"), steps=3,
                               prefix_tokens=mx.array([[7, 8, 9]], mx.int32))
    tokens = mx.array([[1, 2, 3]], mx.int32)
    before = runtime.correctness(tokens)
    original = generation.generate_step

    def broken(*args, **kwargs):
        iterator = original(*args, **kwargs)
        try:
            yield next(iterator)
            raise RuntimeError("interrupted generation")
        finally:
            iterator.close()

    with monkeypatch.context() as patch:
        patch.setattr(generation, "generate_step", broken)
        with pytest.raises(RuntimeError, match="interrupted generation"):
            runtime(tokens)
    _assert_equal(runtime.correctness(tokens), before)


@pytest.mark.parametrize("steps", [True, 0, -1, 1.5])
def test_generation_count_is_fixed_and_positive(steps):
    with pytest.raises(ValueError, match="positive integer"):
        LibraryInference(_build("llama"), steps=steps)


@pytest.mark.parametrize("kind,scope", [("mamba", "model.backbone.layers.0.mixer@"),
                                        ("llama", "model.model.layers.0.self_attn@")])
def test_library_task_caches_are_recorded_state_holders(kind, scope):
    """The library task builds its cache inside the timed call, so the runtime
    keeps one cache per layer and resets it in place: the tracer registers
    it once, every cache read and write inside the model records as a state
    call, the library loop's own state evaluation stays a recorded boundary
    instead of an unknown array, and nothing is left retained after the call."""
    from autotuner.bind.emit import scope_nodes
    from autotuner.trace import Tracer
    from autotuner.trace.recorder import state_method

    tracer = Tracer()
    tracer.install()
    try:
        runtime = LibraryInference(_build(kind), steps=1)
        trace, out = tracer.trace(runtime, [mx.array([[1, 2, 3, 4]], mx.int32)])
    finally:
        tracer.uninstall()
    assert len(out["tokens"]) == 1 and not trace.python_retained()
    calls = [sc for sc in trace.scope_calls if sc.address.startswith(scope)]
    assert len(calls) >= 2  # the prompt pass and the generated token
    for sc in calls:
        assert any(state_method(n.op) for n in scope_nodes(trace, sc)), sc.address
    if kind == "mamba":
        from autotuner.bind.graph import graph_scope_reason
        assert all(graph_scope_reason(trace, sc) is None for sc in calls)
