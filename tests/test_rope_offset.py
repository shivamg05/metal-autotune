"""Recorded scalar values stay literal; stateful replay falls back at other
positions instead of guessing which constants were reads of cache.offset."""

import mlx.core as mx
import pytest
from pathlib import Path

from autotuner.bind.emit import EmittedWrapper, emit_wrapper
from autotuner.bind.swap import install as swap_install
from autotuner.loop import JobRunner, _load_class, _resolve
from autotuner.measure.session import Session
from mlx_lm.models.cache import make_prompt_cache

FIXTURES = Path(__file__).parent / "fixtures"


def _runner(tmp_path, context=512, model_path=None):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(f"""model: {model_path or FIXTURES / 'rope_context_model.py'}
baseline: plain
workloads:
  - name: decode
    inputs: [{{shape: [1, 1], dtype: int32, high: 16}}]
    context: {context}
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                       clock_pairs=4, session=Session(sleep=lambda s: None))
    runner.load_model()
    runner.trace_workloads()
    return runner


def _layer_wrapper(runner, name="RopeReplay"):
    trace = runner.traces["decode"]
    scope = next(sc for sc in trace.scope_calls if sc.address == "model.layers.0@0")
    return emit_wrapper(trace, scope, [], name)


def test_stateful_wrapper_guards_position_and_preserves_literal(tmp_path):
    runner = _runner(tmp_path, context=512)
    try:
        src = _layer_wrapper(runner).source
    finally:
        runner.tracer.uninstall()
    assert "getattr(a1, 'offset', None) == 512" in src
    assert "offset=512" in src
    assert "offset=a1.offset" not in src


def _decode(inner, wrapper, n=4, context=512):
    """A real generation loop on the fixture: prefill `context` tokens, then
    step from the recorded position, letting the offset advance. Returns
    per-step outputs."""
    cache = make_prompt_cache(inner)
    mx.eval(inner(mx.random.randint(0, 16, (1, context), key=mx.random.key(1)), cache=cache))
    target = _resolve(inner, "layers.0")
    if wrapper is not None:
        swap_install(inner, "layers.0", _load_class(wrapper)(target, {}))
    outputs = []
    for i in range(n):
        y = inner(mx.random.randint(0, 16, (1, 1), key=mx.random.key(100 + i)), cache=cache)
        mx.eval(y)
        outputs.append(y)
    if wrapper is not None:
        swap_install(inner, "layers.0", target)
    return outputs


@pytest.mark.parametrize("mode", ["live", "constant", "read_before_update"])
def test_replay_matches_advancing_decode_and_uses_fallback(tmp_path, mode):
    source = (FIXTURES / "rope_context_model.py").read_text()
    if mode == "constant":
        source = source.replace("offset = cache.offset if cache is not None else 0",
                                "offset = 512 if cache is not None else 0")
    elif mode == "read_before_update":
        source = source.replace(
            "        q = mx.fast.rope(",
            "        ctx = cache.update_and_fetch(x) if cache is not None else x\n"
            "        q = mx.fast.rope(")
        source = source.replace(
            "        ctx = cache.update_and_fetch(qf) if cache is not None else qf\n", "")
    model_path = tmp_path / "model.py"
    model_path.write_text(source)
    runner = _runner(tmp_path, model_path=model_path)
    try:
        replay = _layer_wrapper(runner)
    finally:
        runner.tracer.uninstall()
    inner = runner.model.model
    reference = _decode(inner, None)
    # Instrument the original branch so equality cannot hide frozen replay.
    instrumented = EmittedWrapper(
        replay.class_name, replay.scope_path,
        replay.source.replace("return self.wrapped(",
                              "self.fallbacks.append(True); return self.wrapped("), [], [])
    cls = _load_class(instrumented)
    cls.fallbacks = []
    target = _resolve(inner, "layers.0")
    cache = make_prompt_cache(inner)
    mx.eval(inner(mx.random.randint(0, 16, (1, 512), key=mx.random.key(1)), cache=cache))
    swap_install(inner, "layers.0", cls(target, {}))
    try:
        for i, expected in enumerate(reference):
            out = inner(mx.random.randint(0, 16, (1, 1), key=mx.random.key(100 + i)), cache=cache)
            mx.eval(out)
            assert bool(mx.array_equal(expected, out))
        assert len(cls.fallbacks) == 3
    finally:
        swap_install(inner, "layers.0", target)


def test_a_bare_prefill_keeps_the_zero_offset_literal(tmp_path):
    """No context means no cache at all; rope's offset is a genuine literal 0
    the model computed, not a read off a cache, so it stays literal."""
    manifest = tmp_path / "m.yaml"
    manifest.write_text(f"""model: {FIXTURES / 'rope_context_model.py'}
baseline: plain
workloads:
  - name: prefill
    inputs: [{{shape: [1, 4], dtype: int32, high: 16}}]
""")
    runner = JobRunner(manifest, tmp_path / "work", judge_factory=lambda region: None,
                       clock_pairs=4, session=Session(sleep=lambda s: None))
    runner.load_model()
    runner.trace_workloads()
    try:
        trace = runner.traces["prefill"]
        scope = next(sc for sc in trace.scope_calls if sc.address == "layers.0@0")
        src = emit_wrapper(trace, scope, [], "Prefill").source
    finally:
        runner.tracer.uninstall()
    assert "offset=0" in src and ".offset" not in src


def test_stateful_variants_include_position_in_route(tmp_path):
    from autotuner.bind.emit import emit_wrapper_variants

    variants = []
    for context in (4, 8):
        folder = tmp_path / str(context)
        folder.mkdir()
        runner = _runner(folder, context=context)
        try:
            trace = runner.traces["decode"]
            scope = next(sc for sc in trace.scope_calls if sc.address == "model.layers.0@0")
            variants.append((trace, scope, []))
        finally:
            runner.tracer.uninstall()
    source = emit_wrapper_variants(variants, "TwoPositions").source
    assert "getattr(a1, 'offset', None) == 4" in source
    assert "getattr(a1, 'offset', None) == 8" in source
    assert "offset=4" in source and "offset=8" in source
