"""Generated replay wrappers, identity certification, the literal
retrace check, shape fallback, and rollback, on real fixtures."""

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from autotuner.bind.certify import certify_identity, screen_scope
from autotuner.bind.emit import MODULE_HEADER, Splice, emit_wrapper
from autotuner.bind.verify import verify_retrace
from autotuner.trace import Tracer
from tests.conftest import current_tracer, tracer_for_module
from autotuner_runtime.kernels import KernelSpec
from autotuner_runtime.swap import install, uninstall

# whole jobs and live models: minutes, not seconds
pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures"

_module_tracer = tracer_for_module()


def tracer() -> Tracer:
    return current_tracer()


def load_fixture(name: str):
    tracer()
    spec = importlib.util.spec_from_file_location(f"fixture_b_{name}", FIXTURES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # a compiled function in it is named by this import path
    spec.loader.exec_module(mod)
    return mod.build()


def build_wrapper_class(emitted):
    ns = {}
    exec(compile(MODULE_HEADER + emitted.source, "<generated>", "exec"), ns)
    return ns[emitted.class_name]


def scope_call_at(trace, address):
    return next(sc for sc in trace.scope_calls if sc.address == address)


def flat(outs):
    arrays = []

    def walk(o):
        if isinstance(o, mx.array):
            arrays.append(o)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(outs)
    mx.eval(arrays)
    return arrays


ADD_KERNEL = KernelSpec(
    kernel_id="k_add_residual", name="bind_test_add",
    input_names=("a", "b"), output_names=("out",),
    source="uint i = thread_position_in_grid.x;\nout[i] = a[i] + b[i];",
    grid=("in0.shape[0] * in0.shape[1]", "1", "1"),
    threadgroup=("min(in0.shape[0] * in0.shape[1], 256)", "1", "1"),
    output_shapes=(("in0.shape[0]", "in0.shape[1]"),),
    output_dtypes=("float32",),
)


def test_identity_wrapper_is_bitwise_invisible():
    """The whole bind wager: replaying a scope's recorded ops with no kernel
    change must be indistinguishable from the original module."""
    model = load_fixture("repeated_layers")
    x = mx.random.normal((4, 16), key=mx.random.key(0))
    trace, outs = tracer().trace(model, [x])
    baseline = flat(model(x))

    emitted = emit_wrapper(trace, scope_call_at(trace, "layers.1@0"), [], "IdLayer1")
    cls = build_wrapper_class(emitted)
    original = model.layers[1]
    occupant = install(model, "layers.1", cls(original, {}))
    try:
        got = flat(model(x))
        assert all(mx.array_equal(g, b).item() for g, b in zip(got, baseline))
        retrace, _ = tracer().trace(model, [x])
        assert [n.op for n in retrace.nodes] == [n.op for n in trace.nodes]
    finally:
        uninstall(model, "layers.1", occupant)
    restored = flat(model(x))
    assert all(mx.array_equal(r, b).item() for r, b in zip(restored, baseline))


def test_certify_identity_harness():
    model = load_fixture("repeated_layers")
    x1 = mx.random.normal((4, 16), key=mx.random.key(1))
    x2 = mx.random.normal((9, 16), key=mx.random.key(2))
    trace, _ = tracer().trace(model, [x1])
    emitted = emit_wrapper(trace, scope_call_at(trace, "layers.2@0"), [], "IdLayer2")
    cls = build_wrapper_class(emitted)

    def install_cb(wrapper):
        occupant = install(model, "layers.2", wrapper)
        return lambda: uninstall(model, "layers.2", occupant)

    result = certify_identity(
        build_wrapper=lambda: cls(model.layers[2], {}),
        install=install_cb,
        runs=[lambda: model(x1), lambda: model(x2)],
    )
    assert result.ok, result.reason


def test_ship_kernel_through_generated_wrapper_and_verify_retrace():
    """A hand kernel for the residual add ships through a generated wrapper:
    outputs bitwise (elementwise add preserves order), the retrace loses the
    member op and gains one custom dispatch feeding the same consumers, and
    rollback restores the original module."""
    model = load_fixture("repeated_layers")
    x = mx.random.normal((4, 16), key=mx.random.key(3))
    trace, _ = tracer().trace(model, [x])
    baseline = flat(model(x))

    add_node = next(
        n for n in trace.nodes
        if n.op == "array.__add__" and n.module_address == "layers.1@0"
    )
    splice = Splice(
        kernel=ADD_KERNEL,
        start_seq=add_node.seq, end_seq=add_node.seq,
        input_ids=tuple(add_node.in_arrays),
        output_ids=tuple(add_node.out_arrays),
        fingerprint="r_add",
    )
    emitted = emit_wrapper(trace, scope_call_at(trace, "layers.1@0"), [splice], "SpliceLayer1")
    cls = build_wrapper_class(emitted)
    occupant = install(model, "layers.1", cls(model.layers[1], {ADD_KERNEL.kernel_id: ADD_KERNEL}))
    try:
        got = flat(model(x))
        assert all(mx.array_equal(g, b).item() for g, b in zip(got, baseline))

        retrace, _ = tracer().trace(model, [x])
        report = verify_retrace(
            trace, retrace,
            cut_spans=[(add_node.seq, add_node.seq)],
            expected_kernel_ids=[ADD_KERNEL.kernel_id],
        )
        assert report.ok, report.reasons
        assert "array.__add__" in [n.op for n in retrace.nodes]  # the other layers' adds remain
        assert sum(1 for n in retrace.nodes if n.op == "custom_kernel") == 1
    finally:
        uninstall(model, "layers.1", occupant)
    restored, _ = tracer().trace(model, [x])
    assert [n.op for n in restored.nodes] == [n.op for n in trace.nodes]


def test_fallback_path_replays_original_ops():
    """A shape-specialized kernel's uncovered shapes fall through to the
    original op sequence carried in the generated code."""
    model = load_fixture("repeated_layers")
    x4 = mx.random.normal((4, 16), key=mx.random.key(4))
    x7 = mx.random.normal((7, 16), key=mx.random.key(5))
    trace, _ = tracer().trace(model, [x4])
    base4, base7 = flat(model(x4)), flat(model(x7))

    add_node = next(
        n for n in trace.nodes
        if n.op == "array.__add__" and n.module_address == "layers.1@0"
    )
    specialized = KernelSpec(**{
        **{k: getattr(ADD_KERNEL, k) for k in ADD_KERNEL.__dataclass_fields__},
        "kernel_id": "k_add_batch4", "fallback_predicate": "in0.shape[0] != 4",
    })
    splice = Splice(
        kernel=specialized,
        start_seq=add_node.seq, end_seq=add_node.seq,
        input_ids=tuple(add_node.in_arrays),
        output_ids=tuple(add_node.out_arrays),
    )
    emitted = emit_wrapper(trace, scope_call_at(trace, "layers.1@0"), [splice], "FallbackLayer1")
    cls = build_wrapper_class(emitted)
    occupant = install(model, "layers.1", cls(model.layers[1], {specialized.kernel_id: specialized}))
    try:
        assert all(mx.array_equal(g, b).item() for g, b in zip(flat(model(x4)), base4))
        assert all(mx.array_equal(g, b).item() for g, b in zip(flat(model(x7)), base7))
        retrace4, _ = tracer().trace(model, [x4])
        assert any(n.op == "custom_kernel" for n in retrace4.nodes)
        retrace7, _ = tracer().trace(model, [x7])
        assert not any(n.op == "custom_kernel" for n in retrace7.nodes)
        assert [n.op for n in retrace7.nodes] == [n.op for n in trace.nodes]
    finally:
        uninstall(model, "layers.1", occupant)


def test_wrapper_hands_unrecorded_shapes_to_the_original_module():
    """A replay is exact only at the recorded shapes, so a wrapper called
    at any other shape must run the wrapped module itself: same outputs, no
    custom dispatch, and the original ops back in the record."""
    model = load_fixture("repeated_layers")
    x4 = mx.random.normal((4, 16), key=mx.random.key(4))
    x7 = mx.random.normal((7, 16), key=mx.random.key(5))
    trace, _ = tracer().trace(model, [x4])
    base7 = flat(model(x7))
    add_node = next(
        n for n in trace.nodes
        if n.op == "array.__add__" and n.module_address == "layers.1@0"
    )
    splice = Splice(
        kernel=ADD_KERNEL, start_seq=add_node.seq, end_seq=add_node.seq,
        input_ids=tuple(add_node.in_arrays), output_ids=tuple(add_node.out_arrays),
    )
    emitted = emit_wrapper(trace, scope_call_at(trace, "layers.1@0"), [splice], "GuardedLayer1")
    assert "a0.shape == (4, 16)" in emitted.source
    cls = build_wrapper_class(emitted)
    occupant = install(model, "layers.1", cls(model.layers[1], {ADD_KERNEL.kernel_id: ADD_KERNEL}))
    try:
        retrace4, _ = tracer().trace(model, [x4])
        assert any(n.op == "custom_kernel" for n in retrace4.nodes)
        assert all(mx.array_equal(g, b).item() for g, b in zip(flat(model(x7)), base7))
        retrace7, _ = tracer().trace(model, [x7])
        assert not any(n.op == "custom_kernel" for n in retrace7.nodes)
        assert [n.op for n in retrace7.nodes] == [n.op for n in trace.nodes]
    finally:
        uninstall(model, "layers.1", occupant)


def test_screen_rejects_unreplayable_scopes():
    cases = {
        "cache_retention": ((4, 16), "python-retained"),
        "opaque_submodule": ((4, 8), "compiled call the harness cannot name"),
        "data_branch": ((4, 8), "evaluates mid-call"),
    }
    for name, (shape, needle) in cases.items():
        model = load_fixture(name)
        x = mx.random.normal(shape, key=mx.random.key(6))
        trace, _ = tracer().trace(model, [x])
        root_stack = trace.nodes[0].module_stack[:1]
        reason = screen_scope(trace, root_stack)
        assert reason is not None and needle in reason, (name, reason)


def test_scope_around_a_named_compiled_call_replays_bitwise():
    """The model's own compiled section is one opaque call, but a scope that
    contains it still hosts a wrapper: the wrapper calls the compiled function
    by import path, exactly as the model does."""
    model = load_fixture("compiled_submodule")
    x = mx.random.normal((4, 8), key=mx.random.key(8))
    trace, _ = tracer().trace(model, [x])
    part_stack = next(sc.stack for sc in trace.scope_calls if sc.address == "part@0")
    assert screen_scope(trace, part_stack) is None
    emitted = emit_wrapper(trace, scope_call_at(trace, "part@0"), [], "IdPart")
    assert "_kernels.imported('fixture_b_compiled_submodule.fast_tanh')" in emitted.source
    cls = build_wrapper_class(emitted)

    def install_cb(wrapper):
        occupant = install(model, "part", wrapper)
        return lambda: uninstall(model, "part", occupant)

    result = certify_identity(build_wrapper=lambda: cls(model.part, {}), install=install_cb,
                              runs=[lambda: model(x)])
    assert result.ok, result.reason


def test_scope_around_a_named_custom_kernel_replays_bitwise():
    """The model's own custom kernel is one opaque call, but a scope that
    contains it still hosts a wrapper: the wrapper rebuilds the captured
    definition and uses the recorded launch, exactly as the model does."""
    model = load_fixture("kernel_submodule")
    x = mx.random.normal((4, 8), key=mx.random.key(8))
    trace, _ = tracer().trace(model, [x])
    part_stack = next(sc.stack for sc in trace.scope_calls if sc.address == "part@0")
    assert screen_scope(trace, part_stack) is None
    emitted = emit_wrapper(trace, scope_call_at(trace, "part@0"), [], "IdPart")
    assert "_captured(" in emitted.source
    cls = build_wrapper_class(emitted)

    def install_cb(wrapper):
        occupant = install(model, "part", wrapper)
        return lambda: uninstall(model, "part", occupant)

    result = certify_identity(build_wrapper=lambda: cls(model.part, {}), install=install_cb,
                              runs=[lambda: model(x)])
    assert result.ok, result.reason


def test_screen_accepts_replayable_scope():
    model = load_fixture("repeated_layers")
    x = mx.random.normal((4, 16), key=mx.random.key(7))
    trace, _ = tracer().trace(model, [x])
    layer_stack = next(sc.stack for sc in trace.scope_calls if sc.address == "layers.0@0")
    assert screen_scope(trace, layer_stack) is None
