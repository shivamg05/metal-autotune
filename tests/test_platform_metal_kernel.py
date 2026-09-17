"""Pinned platform facts for mx.fast.metal_kernel, graduated from spike_04
(delivery) and spike_06 (metal_kernel behavior). Each docstring names the
design argument the fact protects, so an mlx upgrade that changes the
behavior fails here loudly instead of silently invalidating the harness.

Behavior only, never timing; spikes/out/logs/ stay the source for numbers.
"""

import mlx.core as mx
import pytest

COPY_SRC = "uint elem = thread_position_in_grid.x;\nout[elem] = inp[elem];\n"


def kernel(name, source, **kwargs):
    return mx.fast.metal_kernel(
        name=name, input_names=["inp"], output_names=["out"], source=source, **kwargs)


def run(k, inputs, shapes, dtypes, grid, tg, **kwargs):
    outs = k(inputs=inputs, output_shapes=shapes, output_dtypes=dtypes,
             grid=grid, threadgroup=tg, **kwargs)
    mx.eval(outs)
    return outs


def test_kernel_object_exposes_nothing_but_its_call():
    """Design argument: the tracer records a model's own custom kernel by
    handing the model a stand-in from the mx.fast.metal_kernel factory,
    because the object the factory returns has no source, name, or input
    names to read back and no attribute to hook."""
    k = kernel("pin_surface", COPY_SRC)
    assert [a for a in dir(k) if not a.startswith("__")] == []


def test_construction_and_call_signature_keyword_only():
    """The harness owns every kernel call site: generated wrappers and the
    sandbox worker hardcode exactly this construction and keyword-only call
    surface, so a renamed or repositioned parameter must fail here, not
    silently inside generated artifact code."""
    src = "uint elem = thread_position_in_grid.x;\nout[elem] = static_cast<T>(inp[elem]) * PIN_TWO;\n"
    k = mx.fast.metal_kernel(
        name="pin_call_sig",
        input_names=["inp"],
        output_names=["out"],
        source=src,
        header="#define PIN_TWO 2.0f\n",
        ensure_row_contiguous=True,
        atomic_outputs=False,
        compile_options={"math_mode": "safe"},
    )
    a = mx.arange(8, dtype=mx.float32)
    with pytest.raises(TypeError):
        k([a], [(8,)], [mx.float32], (8, 1, 1), (8, 1, 1))
    (out,) = run(k, [a], [(8,)], [mx.float32], (8, 1, 1), (8, 1, 1),
                 template=[("T", mx.float32)], init_value=0.0, verbose=False)
    assert mx.array_equal(out, a * 2).item()


def test_grid_is_total_threads():
    """Scaffold launch sizing and the judge's grid guidance assume grid means
    TOTAL THREADS (dispatchThreads semantics): 50 threads under a 32-wide
    threadgroup is only expressible that way (dispatchThreadgroups would
    report threads_per_grid.x == 1600)."""
    src = ("uint elem = thread_position_in_grid.x;\n"
           "out[elem] = (float)elem;\n"
           "tpg[0] = (float)threads_per_grid.x;\n")
    k = mx.fast.metal_kernel(name="pin_grid", input_names=["inp"],
                             output_names=["out", "tpg"], source=src)
    a = mx.zeros((50,), dtype=mx.float32)
    out, tpg = run(k, [a], [(50,), (1,)], [mx.float32, mx.float32],
                   (50, 1, 1), (32, 1, 1))
    assert mx.array_equal(out, mx.arange(50, dtype=mx.float32)).item()
    assert tpg[0].item() == 50.0


def test_compile_error_is_runtimeerror_at_eval_only():
    """The ladder's compile gate works by evaluating a probe output:
    construction and call both accept broken source silently, so a gate that
    skipped the eval would pass a kernel that cannot build."""
    k = kernel("pin_broken", "uint elem = thread_position_in_grid.x;\nPIN_BROKEN_SYMBOL;\n")
    a = mx.zeros((8,), dtype=mx.float32)
    out = k(inputs=[a], output_shapes=[(8,)], output_dtypes=[mx.float32],
            grid=(8, 1, 1), threadgroup=(8, 1, 1))
    with pytest.raises(RuntimeError):
        mx.eval(out)


def test_non_c_identifier_name_fails_like_a_compile_error():
    """The kernel name is pasted into the generated MSL signature, so static
    checks must reject non-C-identifier names: construction and call accept a
    hyphenated name silently and the failure is the same RuntimeError at eval
    that a broken body produces."""
    k = kernel("pin-bad-name", COPY_SRC)
    a = mx.zeros((8,), dtype=mx.float32)
    out = k(inputs=[a], output_shapes=[(8,)], output_dtypes=[mx.float32],
            grid=(8, 1, 1), threadgroup=(8, 1, 1))
    with pytest.raises(RuntimeError):
        mx.eval(out)


def test_init_value_nan_poisons_unwritten_outputs():
    """The poison gate exists because init_value=nan makes every unwritten
    output element read back NaN even from a recycled buffer, while written
    elements pass through exactly; a kernel that skips elements cannot pass
    by inheriting plausible stale memory."""
    src = "uint elem = thread_position_in_grid.x;\nif (elem < 512) { out[elem] = inp[elem]; }\n"
    k = kernel("pin_poison", src)
    a = mx.random.normal((1024,), key=mx.random.key(0)).astype(mx.float32)
    mx.eval(a)
    for _ in range(3):
        # dirty the buffer pool so a recycled non-NaN buffer would show through
        garbage = mx.full((1024,), 7.0, dtype=mx.float32)
        mx.eval(garbage)
        del garbage
        (out,) = run(k, [a], [(1024,)], [mx.float32], (1024, 1, 1), (256, 1, 1),
                     init_value=float("nan"))
        assert mx.array_equal(out[:512], a[:512]).item()
        assert mx.all(mx.isnan(out[512:])).item()


def test_shape_stride_ndim_buffers_auto_injected():
    """Naive scaffold lowering indexes strided inputs through inp_shape,
    inp_strides, and inp_ndim; referencing those names must auto-inject
    buffers carrying the right values or every non-flat scaffold breaks."""
    src = ("meta[0] = (float)inp_ndim;\n"
           "meta[1] = (float)inp_shape[0];\n"
           "meta[2] = (float)inp_shape[1];\n"
           "meta[3] = (float)inp_strides[0];\n"
           "meta[4] = (float)inp_strides[1];\n")
    k = mx.fast.metal_kernel(name="pin_shapes", input_names=["inp"],
                             output_names=["meta"], source=src)
    a = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    (meta,) = run(k, [a], [(5,)], [mx.float32], (1, 1, 1), (1, 1, 1))
    assert [v.item() for v in meta] == [2.0, 3.0, 4.0, 4.0, 1.0]


def test_ensure_row_contiguous_true_copies_strided_views():
    """Scaffolds index flat and rely on ensure_row_contiguous=True handing the
    kernel a row-major copy of any strided view, so a transposed input reads
    back in logical order with no stride arithmetic in the kernel."""
    a = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    at = mx.transpose(a)
    k = kernel("pin_rc_true", COPY_SRC, ensure_row_contiguous=True)
    (out,) = run(k, [at], [(12,)], [mx.float32], (12, 1, 1), (12, 1, 1))
    assert mx.array_equal(out, mx.reshape(at, (12,))).item()


def test_ensure_row_contiguous_false_reads_raw_memory_silently():
    """ensure_row_contiguous=False with flat indexing reads the raw buffer of
    a strided view and raises nothing: silent wrongness that only the
    ladder's correctness gates can catch. The gate design counts on this
    failure mode staying silent rather than erroring earlier."""
    a = mx.arange(12, dtype=mx.float32).reshape(3, 4)
    at = mx.transpose(a)
    k = kernel("pin_rc_false", COPY_SRC, ensure_row_contiguous=False)
    (out,) = run(k, [at], [(12,)], [mx.float32], (12, 1, 1), (12, 1, 1))
    assert mx.array_equal(out, mx.reshape(a, (12,))).item()
    assert not mx.array_equal(out, mx.reshape(at, (12,))).item()


def test_same_name_different_source_independent_in_process():
    """The loop compiles many hypothesis kernels for one region under one
    stable name; two same-name kernels with different source must stay
    independent within a process, or a name-keyed cache would serve a stale
    binary and the harness would time the wrong kernel."""
    k1 = kernel("pin_samename", "uint elem = thread_position_in_grid.x;\nout[elem] = 1.0f;\n")
    k2 = kernel("pin_samename", "uint elem = thread_position_in_grid.x;\nout[elem] = 2.0f;\n")
    a = mx.zeros((4,), dtype=mx.float32)

    def first(k):
        (out,) = run(k, [a], [(4,)], [mx.float32], (4, 1, 1), (4, 1, 1))
        return out[0].item()

    assert [first(k1), first(k2), first(k1), first(k2)] == [1.0, 2.0, 1.0, 2.0]


def test_precise_namespace_gives_library_bits_not_math_mode():
    """Scaffold fidelity to library bits comes from metal::precise::
    namespacing, not math_mode: precise::exp matches mx.exp bitwise under
    both safe and fast math, plain metal::exp does not. Naive lowering uses
    precise:: because of this, and math_mode alone cannot buy it back."""
    a = mx.random.normal((1024,), key=mx.random.key(1)).astype(mx.float32)
    mx.eval(a)
    ref_bits = mx.view(mx.exp(a), mx.uint32)

    def exp_bits(tag, call, mode):
        src = f"uint elem = thread_position_in_grid.x;\nout[elem] = {call}(inp[elem]);\n"
        k = kernel(f"pin_exp_{tag}", src, compile_options={"math_mode": mode})
        (out,) = run(k, [a], [(1024,)], [mx.float32], (1024, 1, 1), (256, 1, 1))
        return mx.view(out, mx.uint32)

    assert mx.array_equal(exp_bits("precise_safe", "metal::precise::exp", "safe"), ref_bits).item()
    assert mx.array_equal(exp_bits("precise_fast", "metal::precise::exp", "fast"), ref_bits).item()
    assert not mx.array_equal(exp_bits("plain_safe", "metal::exp", "safe"), ref_bits).item()
