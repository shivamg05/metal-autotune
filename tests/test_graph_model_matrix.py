"""Real model calculations with graph-installed Metal, independent of search luck.

Small architectures keep these correctness checks fast. They do not establish
production-model speedups. Each case starts before custom kernel construction.
"""
import subprocess
import sys

import pytest


def exercise(family, operation, context):
    import mlx.core as mx
    from autotuner.trace import Tracer
    from autotuner.bind.certify import find_scope_call
    from autotuner.bind.emit import MODULE_HEADER, Splice
    from autotuner.bind.graph import emit_graph_wrapper_variants, graph_scope_reason
    from autotuner.regions.build import build_stretches, is_view
    from autotuner.regions.fingerprint import fingerprint
    from autotuner.scaffold import build_scaffold
    from autotuner.artifact.validate import check_outputs
    from autotuner.e2e import share_weights
    from autotuner_runtime.swap import install, resolve_value
    from autotuner_runtime.state import context_step, correctness_call
    from autotuner_runtime.kernels import KernelSpec
    from tests.test_library_inference import _build

    tracer = Tracer()
    tracer.install()
    try:
        if family == "flux":
            import models.flux2_4b as definition
            # Same denoiser implementation and block types, smaller dimensions.
            definition.HEADS, definition.HEAD_DIM, definition.HIDDEN = 2, 32, 64
            definition.MLP_INNER = 128
            definition.DOUBLE_BLOCKS = definition.SINGLE_BLOCKS = 1
            definition.IN_CHANNELS, definition.TXT_DIM = 8, 64
            definition.AXES_DIM = (8, 8, 8, 8)
            original, candidate = definition.Flux2(), definition.Flux2()
            inputs = [mx.random.normal((1, 3, 8)), mx.random.normal((1, 2, 64)), mx.array([.5])]
        else:
            original, candidate = _build(family), _build(family)
            inputs = [mx.array([[1, 2, 3]], mx.int32)]
        share_weights(original, candidate)
        if context >= 0:
            prefix = mx.array([[i % 64 for i in range(context)]], mx.int32)
            inputs = [mx.array([[11]], mx.int32)] if context else inputs
            original = context_step(original, context, prefix, inputs)
            candidate = context_step(candidate, context, prefix, inputs)
        trace, _ = tracer.trace(candidate, inputs)
    finally:
        tracer.uninstall()

    eligible = []
    for span in build_stretches(trace, "case"):
        nodes = trace.nodes[span.start_seq:span.end_seq + 1]
        scope = find_scope_call(trace, span.scope_stack)
        if scope is None or not scope.address.rsplit("@", 1)[0] or graph_scope_reason(trace, scope):
            continue
        ops = [n.op for n in nodes]
        compute = [n.op for n in nodes if not is_view(n)]  # a weight transpose rides along
        if operation == "gemm" and len(compute) == 1 and "matmul" in compute[0]:
            eligible.append((span, scope))
        elif operation == "attention" and len(nodes) == 1 and "scaled_dot_product_attention" in ops[0]:
            eligible.append((span, scope))
        elif operation == "fusion" and len(nodes) >= 2 and all(
                op in {"array.__mul__", "array.__add__", "array.__sub__", "mx.multiply", "mx.add"} for op in ops):
            eligible.append((span, scope))
    assert eligible, (family, operation, context, "no graph-installable target")
    span, scope = eligible[0]
    # A rule replaces every identical occurrence in its scope, so the loop
    # splices all copies of a region there together; a token loop repeats one.
    copies = [s for s, sc in eligible if sc.address == scope.address
              and fingerprint(trace, s) == fingerprint(trace, span)]
    if operation == "attention":
        node = trace.nodes[span.start_seq]
        scale = node.scalar_args["kwargs"]["scale"]
        # A slow but real attention kernel exercises replacing this primitive.
        # One thread computes one output, with a stable softmax over key rows.
        kernel = KernelSpec(
            "matrix_attention", "matrix_attention", ("q", "k", "v"), ("out",),
            f"""uint i=thread_position_in_grid.x;
uint D=q_shape[3], Q=q_shape[2], K=k_shape[2], H=q_shape[1], HK=k_shape[1], V=v_shape[3];
uint d=i%V, row=(i/V)%Q, h=(i/(V*Q))%H, b=i/(V*Q*H), hk=h/(H/HK);
float top=-INFINITY, total=0.0f, result=0.0f;
for(uint s=0;s<K;s++){{float score=0.0f;
for(uint j=0;j<D;j++) score+=float(q[((b*H+h)*Q+row)*D+j])*float(k[((b*HK+hk)*K+s)*D+j]);
top=metal::max(top,score*{scale!r}f);}}
for(uint s=0;s<K;s++){{float score=0.0f;
for(uint j=0;j<D;j++) score+=float(q[((b*H+h)*Q+row)*D+j])*float(k[((b*HK+hk)*K+s)*D+j]);
float p=metal::exp(score*{scale!r}f-top); total+=p;
result+=p*float(v[((b*HK+hk)*K+s)*V+d]);}}
out[i]=result/total;""",
            grid=("in0.shape[0]*in0.shape[1]*in0.shape[2]*in2.shape[3]", "1", "1"),
            threadgroup=("32", "1", "1"),
            output_shapes=(("in0.shape[0]", "in0.shape[1]", "in0.shape[2]", "in2.shape[3]"),),
            output_dtypes=(node.out_specs[0][1],))
        assert len(span.input_ids) == 3, "this attention fixture has no mask"
    else:
        # Copies can differ in shape; the loop hands every instance to the
        # lowerer so those dims stay symbolic instead of baked literals.
        shapes = [[trace.span_specs(s.start_seq, s.end_seq)[a][0] for a in s.input_ids] for s in copies]
        kernel = build_scaffold(trace, span, shapes)
    splices = [Splice(kernel, s.start_seq, s.end_seq, s.input_ids, s.output_ids) for s in copies]
    emitted = emit_graph_wrapper_variants([(trace, scope, splices)], "MatrixReplacement")
    namespace = {}
    exec(MODULE_HEADER + emitted.source, namespace)
    wrapper = namespace[emitted.class_name](resolve_value(candidate, emitted.scope_path), {kernel.kernel_id: kernel})
    install(candidate, emitted.scope_path, wrapper)
    # This checks delivery, not the starter's numerics: on these fp32 fixtures
    # the stitched quantized matmul and the naive elementwise kernel differ
    # from the library in the last bit (1 ulp), which the ladder's exact gate
    # judges in a real job. Tolerance here, bit-exactness there.
    for repeat in range(3):
        result = check_outputs(lambda: correctness_call(original, inputs),
                               lambda: correctness_call(candidate, inputs), "model", exact=False)
        assert result["passed"], (family, operation, context, result)
    assert any(row["hits"] for call in wrapper._graph_evidence for row in call)


@pytest.mark.parametrize("family,operation,context", [
    ("llama", "gemm", 0), ("llama", "gemm", 5),
    ("qwen", "gemm", 0), ("qwen", "gemm", 5),
    ("mamba", "gemm", 0), ("mamba", "gemm", 5),
    ("flux", "gemm", -1), ("flux", "attention", -1), ("flux", "fusion", -1),
])
def test_real_model_graph_replacements(family, operation, context):
    result = subprocess.run([sys.executable, "-c",
        "from tests.test_graph_model_matrix import exercise; import sys; "
        "exercise(sys.argv[1],sys.argv[2],int(sys.argv[3]))", family, operation, str(context)],
        text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
