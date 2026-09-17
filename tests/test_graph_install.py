"""Original Python -> rewritten graph -> cached compiled execution."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from autotuner.bind.emit import MODULE_HEADER, NotReplayable, Splice
from autotuner.bind.graph import emit_graph_wrapper_variants, graph_scope_reason
from autotuner.regions.build import build_stretches
from autotuner.scaffold.native import reference_sequence_seed
from autotuner.trace import Tracer
from autotuner_runtime.exact import bitwise_equal
from autotuner_runtime.graph import GraphBindingError
from autotuner_runtime.kernels import KernelSpec, KernelStage


def record(model, arrays):
    tracer = Tracer()
    tracer.install()
    try:
        trace, result = tracer.trace(model, arrays)
    finally:
        tracer.uninstall()
    return trace, result


def wrap(model, trace, splices, address="@0"):
    scope = next(s for s in trace.scope_calls if s.address == address)
    emitted = emit_graph_wrapper_variants([(trace, scope, splices)], "Patched")
    namespace = {}
    exec(MODULE_HEADER + emitted.source, namespace)
    return namespace["Patched"](model, {s.kernel.kernel_id: s.kernel for s in splices})


ADD = KernelSpec("graph_add", "graph_add", ("a", "b"), ("out",),
                 "uint i = thread_position_in_grid.x; out[i] = a[i] + b[i];",
                 grid=("in0.shape[0]", "1", "1"), threadgroup=("32", "1", "1"),
                 output_shapes=(("in0.shape[0]",),), output_dtypes=("float32",))


class Add(nn.Module):
    def __call__(self, x, y):
        return x + y


def test_compiled_graph_replaces_cut_and_stops_rewriting_warm_calls():
    model = Add()
    inputs = [mx.arange(32, dtype=mx.float32), mx.ones((32,))]
    trace, want = record(model, inputs)
    n = trace.nodes[0]
    patched = wrap(model, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)])
    got = patched(*inputs)
    assert bitwise_equal(got, want)
    assert patched._graph_evidence[-1][0]["hits"] == 1
    count = len(patched._graph_evidence)
    for i in range(3):
        changed = [x + i for x in inputs]
        assert bitwise_equal(patched(*changed), model(*changed))
    assert len(patched._graph_evidence) == count
    with patched.validate_graph() as evidence:
        assert bitwise_equal(patched(*inputs), want)
        assert evidence[0][0]["hits"] == 1


def test_uncovered_shape_and_aliases_use_original_module():
    model = Add()
    inputs = [mx.ones((32,)), mx.ones((32,))]
    trace, _ = record(model, inputs)
    n = trace.nodes[0]
    patched = wrap(model, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)])
    assert bitwise_equal(patched(mx.ones((17,)), mx.ones((17,))), mx.full((17,), 2.))
    shared = mx.ones((32,))
    assert bitwise_equal(patched(shared, shared), shared + shared)
    assert bitwise_equal(patched(x=inputs[0], y=inputs[1]), inputs[0] + inputs[1])
    assert patched._graph_evidence == []


def test_live_weight_replacement_does_not_reuse_old_values():
    class Weighted(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.ones((32,))

        def __call__(self, x):
            return x + self.weight

    model = Weighted()
    x = mx.arange(32, dtype=mx.float32)
    trace, _ = record(model, [x])
    n = trace.nodes[0]
    patched = wrap(model, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)])
    assert bitwise_equal(patched(x), model(x))
    count = len(patched._graph_evidence)
    model.weight = mx.full((32,), 9.)
    assert bitwise_equal(patched(x), x + 9.)
    assert len(patched._graph_evidence) == count
    model.weight = mx.array([5.])
    assert bitwise_equal(patched(x), x + 5.)
    assert len(patched._graph_evidence) == count


def test_scalar_constants_match_but_different_constants_do_not():
    class Affine(nn.Module):
        def __call__(self, x):
            return x * 2 + 1

    model = Affine()
    x = mx.arange(32, dtype=mx.float32)
    trace, want = record(model, [x])
    span = next(s for s in build_stretches(trace, "one") if s.start_seq == 0 and s.end_seq == 1)
    seed = reference_sequence_seed(trace, span)
    spec = replace(seed, kernel_id="graph_affine", name="graph_affine", reference_sequence=None,
                   source="uint i = thread_position_in_grid.x; out0[i] = in0[i] * 2.0f + 1.0f;",
                   grid=("32", "1", "1"), threadgroup=("32", "1", "1"))
    patched = wrap(model, trace, [Splice(spec, 0, 1, span.input_ids, span.output_ids)])
    assert bitwise_equal(patched(x), want)


def test_native_mixed_fusion_retains_multiple_live_outputs():
    tracer = Tracer()
    tracer.install()
    try:
        path = Path(__file__).parent / "fixtures/native_fusion.py"
        spec = importlib.util.spec_from_file_location("graph_native_fixture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.build()
        inputs = [mx.arange(32, dtype=mx.float32), mx.ones((32,))]
        trace, want = tracer.trace(model, inputs)
    finally:
        tracer.uninstall()
    span = next(s for s in build_stretches(trace, "one") if s.start_seq == 0 and s.end_seq == 2)
    seed = reference_sequence_seed(trace, span)
    spec = replace(seed, kernel_id="graph_mixed", name="graph_mixed", reference_sequence=None,
                   source="uint i = thread_position_in_grid.x; float v = in0[i] * 2.0f + in1[i]; "
                          "out0[i] = v; out1[i] = v * 2.0f + 1.0f;",
                   grid=("32", "1", "1"), threadgroup=("32", "1", "1"))
    patched = wrap(model, trace, [Splice(spec, 0, 2, span.input_ids, span.output_ids)])
    got = patched(*inputs)
    assert all(bitwise_equal(a, b) for a, b in zip(got, want))
    assert patched._graph_evidence[-1][0]["boundary_outputs"] == 2


def test_argument_anchors_leave_unrelated_matching_shapes_unchanged():
    class TwoAdds(nn.Module):
        def __call__(self, x, y, z):
            a = x + y
            return a, a + z

    model = TwoAdds()
    inputs = [mx.full((32,), float(i)) for i in range(3)]
    trace, _ = record(model, inputs)
    n = trace.nodes[0]
    patched = wrap(model, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)])
    assert all(bitwise_equal(a, b) for a, b in zip(patched(*inputs), model(*inputs)))
    assert patched._graph_evidence[-1][0]["hits"] == 1


def test_stream_changes_build_separate_cached_graphs():
    model = Add()
    inputs = [mx.arange(32, dtype=mx.float32), mx.ones((32,))]
    trace, want = record(model, inputs)
    n = trace.nodes[0]
    patched = wrap(model, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)])
    assert bitwise_equal(patched(*inputs), want)
    count = len(patched._graph_evidence)
    with mx.stream(mx.new_stream(mx.gpu)):
        assert bitwise_equal(patched(*inputs), want)
        assert len(patched._graph_evidence) == count + 1
        assert bitwise_equal(patched(*inputs), want)
        assert len(patched._graph_evidence) == count + 1


def test_python_settings_are_read_when_the_scope_graph_is_built():
    """A module's Python settings are part of the installed calculation, as
    they are for a generated replay: the compiled scope keeps computing what
    was certified, and a warm call spends nothing re-reading them."""
    class Conditional(nn.Module):
        def __init__(self):
            super().__init__()
            self.add = True

        def __call__(self, x, y):
            return x + y if self.add else x - y

    model = Conditional()
    inputs = [mx.arange(32, dtype=mx.float32), mx.ones((32,))]
    trace, _ = record(model, inputs)
    n = trace.nodes[0]
    patched = wrap(model, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)])
    assert bitwise_equal(patched(*inputs), model(*inputs))
    model.add = False
    assert bitwise_equal(patched(*inputs), inputs[0] + inputs[1])
    assert len(patched._graph_evidence) == 1


def test_hidden_python_counter_is_not_frozen_or_advanced_twice():
    class Counter(Add):
        def __init__(self):
            super().__init__()
            self.position = 0

        def __call__(self, x, y):
            self.position += 1
            return x + y + self.position

    model = Counter()
    inputs = [mx.ones((32,)), mx.ones((32,))]
    trace, _ = record(model, inputs)
    model.position = 0
    n = trace.nodes[0]
    patched = wrap(model, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)])
    for position in (1, 2, 3):
        assert bitwise_equal(patched(*inputs), mx.full((32,), float(2 + position)))
        assert model.position == position
    assert patched._graph_fallback_reason == "the module changes Python state inside its call"


def test_tensor_cache_updates_remain_live_across_compiled_calls():
    class Cache:
        def __init__(self):
            self.value = mx.ones((32,))

        def write(self, value):
            self.value = value
            return value

    class Layer(nn.Module):
        def __call__(self, x, cache):
            y = x + cache.value
            cache.write(y)
            return y * 2

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cache = Cache()
            self.layer = Layer()

        def __call__(self, x):
            return self.layer(x, self.cache)

    model = Model()
    x = mx.ones((32,))
    trace, _ = record(model, [x])
    model.cache.value = mx.ones((32,))
    n = next(n for n in trace.nodes if n.op == "array.__add__")
    patched = wrap(model.layer, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)], "layer@0")
    for step in range(1, 5):
        got = patched(x, model.cache)
        assert bitwise_equal(got, mx.full((32,), float((step + 1) * 2)))
        assert bitwise_equal(model.cache.value, mx.full((32,), float(step + 1)))
    assert len(patched._graph_evidence) == 1
    # A fresh independent cache reuses the graph and supplies its own tensors.
    fresh = Cache()
    fresh.value = mx.full((32,), 10.)
    assert bitwise_equal(patched(x, fresh), mx.full((32,), 22.))
    assert bitwise_equal(fresh.value, mx.full((32,), 11.))
    assert len(patched._graph_evidence) == 1


def test_multiple_workloads_share_one_kernel_and_unlisted_shapes_fall_back():
    model = Add()
    variants = []
    for size in (32, 64):
        trace, _ = record(model, [mx.ones((size,)), mx.ones((size,))])
        node = trace.nodes[0]
        variants.append((trace, next(s for s in trace.scope_calls if s.address == "@0"),
                         [Splice(ADD, node.seq, node.seq, node.in_arrays, node.out_arrays)]))
    emitted = emit_graph_wrapper_variants(variants, "MultiShape")
    namespace = {}
    exec(MODULE_HEADER + emitted.source, namespace)
    patched = namespace["MultiShape"](model, {ADD.kernel_id: ADD})
    for size in (32, 64, 33):
        inputs = [mx.arange(size, dtype=mx.float32), mx.ones((size,))]
        assert bitwise_equal(patched(*inputs), model(*inputs))
    assert len(patched._graph_evidence) == 2
    assert patched.KERNEL_IDS == [ADD.kernel_id]


def test_ordered_multi_dispatch_candidate_installs_as_one_graph_cut():
    class Twice(nn.Module):
        def __call__(self, x, y):
            return (x + y) + y

    model = Twice()
    inputs = [mx.arange(32, dtype=mx.float32), mx.ones((32,))]
    trace, want = record(model, inputs)
    span = next(s for s in build_stretches(trace, "one") if s.start_seq == 0 and s.end_seq == 1)
    # Stages name the candidate's own inputs and outputs; each stage kernel
    # keeps the local in0/in1 -> out0 ABI.
    stage = replace(ADD, kernel_id="graph_stage_add", input_names=("in0", "in1"), output_names=("out0",),
                    source="uint i = thread_position_in_grid.x; out0[i] = in0[i] + in1[i];")
    candidate = replace(ADD, kernel_id="graph_stages", stages=(
        KernelStage(("a", "b"), ("tmp0",), stage),
        KernelStage(("tmp0", "b"), ("out",), stage)))
    patched = wrap(model, trace, [Splice(candidate, 0, 1, span.input_ids, span.output_ids)])
    assert bitwise_equal(patched(*inputs), want)
    assert patched._graph_evidence[-1][0]["hits"] == 1


def test_a_cache_with_history_compiles_at_its_recorded_position():
    """A scope stepping a cache that already holds tokens compiles at that
    position, as a replay variant is guarded to it: the position is part of
    the call signature, the compiled call advances it, and the root's rewind
    makes every call the same step."""
    path = Path(__file__).parent / "fixtures/state_call.py"
    spec = importlib.util.spec_from_file_location("graph_offset_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build()
    x = mx.ones((1, 16))
    trace, want = record(model, [x])
    scope = next(s for s in trace.scope_calls if s.address == "layer@0")
    assert graph_scope_reason(trace, scope) is None
    patched = wrap(model.layer, trace, [], "layer@0")
    model.layer = patched
    for _ in range(3):
        assert bitwise_equal(model(x), want) and model.cache.offset == 4
    assert len(patched._graph_evidence) == 1
    model.cache.offset = 6  # a position never recorded runs the original
    with patched.validate_graph() as calls:
        model(x)
    assert calls == [] and not patched._graph_fallbacks


class KVLike:
    """mlx-lm's cache shape: buffers allocated on first use, a position the
    call advances, a method that writes in place, and the cache protocol the
    tracer recognizes an empty cache by."""

    def __init__(self):
        self.keys, self.offset = None, 0

    def is_trimmable(self):
        return True

    def update_and_fetch(self, k):
        prev = self.offset
        if self.keys is None:
            self.keys = mx.zeros((4, 32), k.dtype)
        self.offset += 1
        self.keys[prev:self.offset] = k[None]
        return self.keys[:self.offset]


class Attend(nn.Module):
    def __call__(self, x, cache):
        return cache.update_and_fetch(x + 1.0).sum(axis=0) * 2.0


class Prefill(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = KVLike()
        self.attend = Attend()

    def __call__(self, x):
        return self.attend(x, self.cache)


def test_a_fresh_cache_compiles_and_the_call_advances_it():
    """Prefill: the cache arrives empty at position 0. The compiled scope
    fills it and advances the position exactly as the Python did; a fresh
    cache in the same state reuses the trace; the advanced cache, at a
    position never recorded, runs the original module, which advances it."""
    model = Prefill()
    x = mx.arange(32, dtype=mx.float32)
    trace, want = record(model, [x])
    n = next(n for n in trace.nodes if n.op == "array.__add__")
    add_one = replace(ADD, kernel_id="graph_add_one", name="graph_add_one", input_names=("a",),
                      source="uint i = thread_position_in_grid.x; out[i] = a[i] + 1.0f;")
    patched = wrap(model.attend, trace, [Splice(add_one, n.seq, n.seq, n.in_arrays, n.out_arrays)], "attend@0")
    for _ in range(2):
        cache = KVLike()
        assert bitwise_equal(patched(x, cache), want)
        assert cache.offset == 1 and bitwise_equal(cache.keys[0], x + 1.0)
        assert bitwise_equal(cache.keys[1:], mx.zeros((3, 32)))
    assert len(patched._graph_evidence) == 1
    stepped = model.attend(x, cache)  # the plain module from position 1
    assert cache.offset == 2
    again = KVLike()
    patched(x, again)
    with patched.validate_graph() as calls:
        got = patched(x, again)  # position 1: unrecorded, so the original runs
    assert calls == [] and bitwise_equal(got, stepped) and again.offset == 2
    assert not patched._graph_fallbacks


class Chunked(nn.Module):
    """A cache that appends to a list: the write-back must give the list the
    traced call's length, never the entry length."""

    def __call__(self, x, cache):
        y = x + 1.0
        cache.add(y)
        return y * 2.0


def test_a_holder_list_the_call_grows_is_written_at_the_traced_length():
    class ChunkCache:
        def __init__(self):
            self.chunks = []

        def is_trimmable(self):
            return True

        def add(self, value):
            self.chunks.append(value * 1.0)  # the cache makes what it keeps, as a KV cache does

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cache = ChunkCache()
            self.chunked = Chunked()

        def __call__(self, x):
            return self.chunked(x, self.cache)

    model = Model()
    x = mx.arange(32, dtype=mx.float32)
    trace, want = record(model, [x])
    n = next(n for n in trace.nodes if n.op == "array.__add__")
    add_one = replace(ADD, kernel_id="graph_add_one_list", name="graph_add_one_list", input_names=("a",),
                      source="uint i = thread_position_in_grid.x; out[i] = a[i] + 1.0f;")
    patched = wrap(model.chunked, trace, [Splice(add_one, n.seq, n.seq, n.in_arrays, n.out_arrays)], "chunked@0")
    for _ in range(2):
        cache = ChunkCache()
        assert bitwise_equal(patched(x, cache), want)
        assert len(cache.chunks) == 1 and bitwise_equal(cache.chunks[0], x + 1.0)
    assert len(patched._graph_evidence) == 1


def test_a_compiled_scope_inside_the_harness_compiled_model():
    """A compiled baseline compiles the whole step around the wrapper: the
    outer trace calls the wrapper with tracers, the rewrite still fires, and
    the outputs match the plain model bitwise."""
    model = Add()
    inputs = [mx.arange(32, dtype=mx.float32), mx.ones((32,))]
    trace, want = record(model, inputs)
    n = trace.nodes[0]
    patched = wrap(model, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)])
    whole = mx.compile(lambda *a: patched(*a) * 3.0)
    for _ in range(3):
        assert bitwise_equal(whole(*inputs), want * 3.0)
    assert patched._graph_evidence and patched._graph_evidence[-1][0]["hits"] == 1


def test_compiled_bookkeeping_is_not_model_state():
    from autotuner.trace.walk import snapshot_arrays

    model = Add()
    inputs = [mx.ones((32,)), mx.ones((32,))]
    trace, _ = record(model, inputs)
    n = trace.nodes[0]
    patched = wrap(model, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)])
    mx.eval(patched(*inputs))
    assert snapshot_arrays(patched) == {}


def test_a_problem_in_the_first_trace_never_poisons_later_calls(monkeypatch):
    """MLX caches a trace that returned placeholders; the wrapper must drop
    it, surface the problem once, and trace afresh on the next call, with the
    module's weights intact in between."""
    import autotuner_runtime.kernels as kernels

    class Weighted(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.ones((32,))

        def __call__(self, x):
            return x + self.weight

    model = Weighted()
    x = mx.arange(32, dtype=mx.float32)
    trace, want = record(model, [x])
    n = trace.nodes[0]
    patched = wrap(model, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)])
    real = kernels.try_call

    def broken(*args, **kwargs):
        raise RuntimeError("no metal today")

    monkeypatch.setattr(kernels, "try_call", broken)
    with pytest.raises(RuntimeError, match="no metal today"):
        patched(x)
    assert bitwise_equal(model(x), want)  # no tracer left in the module
    monkeypatch.setattr(kernels, "try_call", real)
    assert bitwise_equal(patched(x), want)
    assert patched._graph_evidence[-1][0]["hits"] == 1


@pytest.mark.parametrize("path", ["fast", "checked"])
def test_a_missed_cut_in_the_trace_falls_back_and_says_why(monkeypatch, path):
    """A rewrite that finds the wrong number of cuts runs the original for
    that signature, records the reason, and never crashes a later call."""
    from autotuner_runtime import graph_native

    if path == "fast":
        model = Add()
        inputs = [mx.arange(32, dtype=mx.float32), mx.ones((32,))]
        trace, want = record(model, inputs)
        n = trace.nodes[0]
        patched = wrap(model, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)])
        call = lambda: patched(*inputs)
        expected = lambda: want
    else:
        class Cache:
            def __init__(self):
                self.value = mx.ones((32,))

            def write(self, value):
                self.value = value
                return value

        class Layer(nn.Module):
            def __call__(self, x, cache):
                y = x + cache.value
                cache.write(y)
                return y * 2

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.cache = Cache()
                self.layer = Layer()

            def __call__(self, x):
                return self.layer(x, self.cache)

        model = Model()
        x = mx.ones((32,))
        trace, _ = record(model, [x])
        model.cache.value = mx.ones((32,))
        n = next(n for n in trace.nodes if n.op == "array.__add__")
        patched = wrap(model.layer, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)], "layer@0")
        steps = [0]
        def call():
            steps[0] += 1
            return patched(x, model.cache)
        expected = lambda: mx.full((32,), float((steps[0] + 1) * 2))
    real = graph_native.rewrite
    monkeypatch.setattr(graph_native, "rewrite", lambda roots, *a, **k: (roots, 0))
    assert bitwise_equal(call(), expected())
    assert patched._graph_fallbacks and "expected 1" in patched._graph_fallbacks[0]
    monkeypatch.setattr(graph_native, "rewrite", real)
    assert bitwise_equal(call(), expected())  # the sentinel is honored, not unpacked
    assert patched._graph_evidence == []


def test_reordered_keyword_arguments_run_the_original():
    """Array references bind by position in the flattened call, so a call
    whose keywords arrive in another order is a different signature."""
    x, y = mx.arange(32, dtype=mx.float32), mx.ones((32,)) * 3

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.add = Add()

        def __call__(self, x, y):
            return self.add(x=x, y=y)

    outer = Outer()
    trace, want = record(outer, [x, y])
    n = trace.nodes[0]
    patched = wrap(outer.add, trace, [Splice(ADD, n.seq, n.seq, n.in_arrays, n.out_arrays)], "add@0")
    assert bitwise_equal(patched(x=x, y=y), want)
    assert patched._graph_evidence[-1][0]["hits"] == 1
    assert bitwise_equal(patched(y=y, x=x), want)
    assert len(patched._graph_evidence) == 1  # the reordered call never reached the graph



def test_a_scope_called_twice_with_one_signature_binds(tmp_path, monkeypatch):
    """A recurrent block calls the same child once per token with identical
    shapes. Those calls share one compiled trace, so bind must count the
    substitutions per traced signature, not once per recorded call."""
    from autotuner.loop import JobRunner, RegionRun
    from autotuner.measure.session import Session
    from tests.test_install import WIN, elementwise

    (tmp_path / "model.py").write_text(
        "import mlx.core as mx\nimport mlx.nn as nn\n\n"
        "class Inner(nn.Module):\n"
        "    def __call__(self, x):\n        return (x * 2) + 1\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n        self.inner = Inner()\n"
        "    def __call__(self, x):\n        return self.inner(self.inner(x))\n\n"
        "def build():\n    return Model()\n")
    (tmp_path / "manifest.yaml").write_text(
        f"model: {tmp_path / 'model.py'}\nbaseline: plain\n"
        "workloads:\n  - name: main\n    inputs: [{shape: [8, 16], dtype: float32}]\n")
    runner = JobRunner(tmp_path / "manifest.yaml", tmp_path / "work", judge_factory=lambda _: None,
                       session=Session(sleep=lambda _: None))
    monkeypatch.setattr(runner, "_model_win", lambda e2e: all(c.passed for c in e2e.checks))
    try:
        runner.load_model()
        runner.trace_workloads()
        region = next(r for r in runner.build_regions() if r.ops == ("array.__mul__",))
        assert region.copies == 2
        kernel = elementwise("twice_mul", "out0[i] = in0[i] * 2.0f;")
        promotion = runner._bind_and_promote(RegionRun(region), kernel, WIN)
        assert promotion, runner.log.rows()[-1]
        assert runner.log.rows()[-2]["kind"] != "bind_failed"
        installation = [r for r in runner.log.rows() if r["kind"] == "installation"]
        assert installation and installation[-1]["method"] == "graph"
    finally:
        runner.tracer.uninstall()
