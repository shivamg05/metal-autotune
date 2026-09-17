"""Native graph substitution preserves value flow, including live side outputs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mlx.core as mx
import pytest

from autotuner_runtime import graph_native as graph


def arrays(shape=(2, 3), dtype=mx.float32):
    return [mx.array([[1, 2, 3], [4, 5, 6]], dtype).reshape(shape),
            mx.array([[6, 5, 4], [3, 2, 1]], dtype).reshape(shape)]


def equal(actual, expected):
    mx.eval(actual, expected)
    for a, b in zip(actual, expected):
        assert mx.array_equal(a, b).item()


def test_replaces_fusion_and_preserves_internal_outside_consumer():
    p, q = arrays()
    pattern = mx.exp(p + q)
    x, y = arrays()
    shared = x + y
    roots = [mx.exp(shared), shared * x]
    result, hits = graph.rewrite(roots, [pattern], [p, q], lambda a: [mx.exp(a[0] + a[1])])
    assert hits == 1
    assert graph.array_id(result[1]) == graph.array_id(roots[1])
    # The external consumer still needs the original shared addition. The
    # replacement has its own addition, so operation sharing correctly differs.
    assert not graph.same_structure(roots, result)[0]
    equal(result, roots)


def test_replaces_multiple_separate_boundary_outputs_together():
    p, q = arrays()
    patterns = [mx.exp(p + q), p * q]
    x, y = arrays()
    first, second = mx.exp(x + y), x * y
    roots = [first + second, first, second]
    calls = []

    def replacement(a):
        calls.append(a)
        return [mx.exp(a[0] + a[1]), a[0] * a[1]]

    result, hits = graph.rewrite(roots, patterns, [p, q], replacement)
    assert hits == len(calls) == 1
    assert graph.same_structure(roots, result)[0]
    equal(result, roots)


def test_boundary_output_with_additional_independent_input():
    p, q = arrays()
    r = mx.array([2.0, 3.0, 4.0])
    x, y = arrays()
    z = mx.array([8.0, 9.0, 10.0])
    result, hits = graph.rewrite([x + y, (x + y) * z], [p + q, (p + q) * r], [p, q, r],
                                lambda a: [a[0] + a[1], (a[0] + a[1]) * a[2]])
    assert hits == 1
    equal(result, [x + y, (x + y) * z])


def test_live_intermediate_is_also_a_boundary_output():
    p, q = arrays()
    intermediate = p + q
    x, y = arrays()
    original = x + y
    result, hits = graph.rewrite([original, mx.exp(original)], [intermediate, mx.exp(intermediate)],
                                [p, q], lambda a: [a[0] + a[1], mx.exp(a[0] + a[1])])
    assert hits == 1
    equal(result, [original, mx.exp(original)])


@pytest.mark.parametrize("kind", ["shape", "dtype", "operator", "axis", "stream", "alias", "scalar"])
def test_nonmatching_case_is_untouched(kind):
    p, q = arrays()
    x, y = arrays()
    pattern, value, parameters = p + q, x + y, [p, q]
    if kind == "shape":
        value = x.T + y.T
    elif kind == "dtype":
        value = x.astype(mx.float16) + y.astype(mx.float16)
    elif kind == "operator":
        value = x - y
    elif kind == "axis":
        pattern, value = mx.sum(p, axis=0), mx.sum(x.T, axis=1)
        parameters = [p]
    elif kind == "stream":
        with mx.stream(mx.new_stream(mx.gpu)):
            value = x + y
    elif kind == "alias":
        pattern, parameters = p + p, [p]
    elif kind == "scalar":
        pattern, value, parameters = p + 1, x + 2, [p]
    result, hits = graph.rewrite([value], [pattern], parameters, lambda _: pytest.fail("unexpected match"))
    assert hits == 0
    assert graph.array_id(result[0]) == graph.array_id(value)


def test_scalar_constants_match_without_becoming_wildcards():
    p, _ = arrays()
    x, _ = arrays()
    roots = [x + 1, x + 2]
    result, hits = graph.rewrite(roots, [p + 1], [p], lambda a: [a[0] + 1])
    assert hits == 1
    equal(result, roots)


def test_evaluated_roots_are_never_rewritten():
    p, q = arrays()
    x, y = arrays()
    root = x + y
    mx.eval(root)
    result, hits = graph.rewrite([root], [p + q], [p, q], lambda _: pytest.fail("evaluated match"))
    assert hits == 0
    assert graph.array_id(result[0]) == graph.array_id(root)


def test_callback_can_decline_specific_live_inputs():
    p, q = arrays()
    x, y = arrays()
    other, _ = arrays()
    roots = [x + y, other + y]
    seen = []

    def replacement(a):
        seen.append(graph.array_id(a[0]))
        return [a[0] + a[1]] if graph.array_id(a[0]) == graph.array_id(x) else None

    result, hits = graph.rewrite(roots, [p + q], [p, q], replacement)
    assert hits == 1
    assert len(seen) == 2
    assert graph.array_id(result[1]) == graph.array_id(roots[1])
    equal(result, roots)


def test_downstream_multioutput_primitive_keeps_every_sibling():
    p, q = arrays(shape=(3, 2))
    x, y = arrays(shape=(3, 2))
    roots = list(mx.linalg.qr(x + y, stream=mx.cpu))
    result, hits = graph.rewrite(roots, [p + q], [p, q], lambda a: [a[0] + a[1]])
    assert hits == 1
    assert graph.same_structure(roots, result)[0]
    equal(result, roots)


def custom_pair(name="graph_pair", add=1):
    kernel = mx.fast.metal_kernel(name=name, input_names=["x"], output_names=["a", "b"],
                                 source=f"uint i = thread_position_in_grid.x; a[i] = x[i] + {add}; b[i] = x[i] * 2;")

    def call(x):
        return kernel(inputs=[x], output_shapes=[x.shape, x.shape], output_dtypes=[x.dtype, x.dtype],
                      grid=(x.size, 1, 1), threadgroup=(32, 1, 1))
    return call


def test_custom_kernel_multioutput_replacement_and_neighbor():
    call = custom_pair()
    p = mx.array([1.0, 2.0, 3.0])
    x = mx.array([4.0, 5.0, 6.0])
    patterns, original = call(p), call(x)
    roots = [original[0] + original[1], original[1]]
    result, hits = graph.rewrite(roots, patterns, [p], lambda a: [a[0] + 1, a[0] * 2])
    assert hits == 1
    equal(result, roots)


@pytest.mark.parametrize("reverse", [False, True])
def test_replacing_one_sibling_does_not_depend_on_root_order(reverse):
    call = custom_pair()
    p = mx.array([1.0, 2.0, 3.0])
    x = mx.array([4.0, 5.0, 6.0])
    patterns, original = call(p), call(x)
    roots = original[::-1] if reverse else original
    result, hits = graph.rewrite(roots, [patterns[0]], [p], lambda a: [a[0] + 1])
    assert hits == 1
    equal(result, roots)


def test_custom_source_change_and_separate_siblings_do_not_match():
    call, different = custom_pair(), custom_pair(add=3)
    p = mx.array([1.0, 2.0, 3.0])
    x = mx.array([4.0, 5.0, 6.0])
    patterns = call(p)
    for roots in [different(x), [call(x)[0], different(x)[1]]]:
        _, hits = graph.rewrite(roots, patterns, [p], lambda _: pytest.fail("different custom source"))
        assert hits == 0


@pytest.mark.parametrize("bad", ["count", "shape", "dtype"])
def test_invalid_callback_outputs_fail_before_execution(bad):
    p, q = arrays()
    x, y = arrays()
    def replacement(a):
        return [] if bad == "count" else [a[0].T if bad == "shape" else a[0].astype(mx.float16)]
    with pytest.raises(ValueError, match="output"):
        graph.rewrite([x + y], [p + q], [p, q], replacement)


def test_compilation_rewrites_once_and_accepts_fresh_input_values():
    p, q = arrays()
    pattern = mx.exp(p + q)
    calls = []
    @mx.compile
    def run(x, y):
        roots, hits = graph.rewrite([mx.exp(x + y)], [pattern], [p, q], lambda a: [mx.exp(a[0] + a[1])])
        calls.append(hits)
        return roots[0]
    x, y = arrays()
    for i in range(4):
        equal([run(x + i, y)], [mx.exp(x + i + y)])
    assert calls == [1]


def test_exported_extension_loads_in_fresh_process(tmp_path):
    package = tmp_path / "graph_native"
    shutil.copytree(Path(graph.__file__).parent, package, ignore=shutil.ignore_patterns("__pycache__"))
    graph.prepare(package)
    program = "import graph_native as g; import mlx.core as m; x=m.array([1]); assert g.array_id(x)>0"
    result = subprocess.run([sys.executable, "-c", program], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    stamp = package / "build.json"
    identity = json.loads(stamp.read_text())
    identity["mlx"] = "incompatible"
    stamp.write_text(json.dumps(identity))
    result = subprocess.run([sys.executable, "-c", program], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0
    assert "does not match" in result.stderr
