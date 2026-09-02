"""The artifact loads in a fresh process with no harness import and
reproduces the patched model's outputs."""

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import mlx.core as mx

from autotuner.artifact.emit import emit_artifact
from autotuner.bind.emit import Splice, emit_wrapper
from autotuner.report import Report
from autotuner.trace import Tracer
from autotuner_runtime.kernels import KernelSpec

FIXTURES = Path(__file__).parent / "fixtures"


def test_artifact_applies_in_fresh_process(tmp_path):
    tracer = Tracer()
    tracer.install()
    try:
        spec = importlib.util.spec_from_file_location("fx", FIXTURES / "repeated_layers.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.build()
        x = mx.random.normal((4, 16), key=mx.random.key(0))
        trace, _ = tracer.trace(model, [x])
    finally:
        tracer.uninstall()

    add_node = next(
        n for n in trace.nodes
        if n.op == "array.__add__" and n.module_address == "layers.3@0"
    )
    kernel = KernelSpec(
        kernel_id="k_art_add", name="artifact_test_add",
        input_names=("a", "b"), output_names=("out",),
        source="uint i = thread_position_in_grid.x;\nout[i] = a[i] + b[i];",
        grid=("in0.shape[0] * in0.shape[1]", "1", "1"),
        threadgroup=("min(in0.shape[0] * in0.shape[1], 256)", "1", "1"),
        output_shapes=(("in0.shape[0]", "in0.shape[1]"),),
        output_dtypes=("float32",),
    )
    splice = Splice(
        kernel=kernel, start_seq=add_node.seq, end_seq=add_node.seq,
        input_ids=tuple(add_node.in_arrays), output_ids=tuple(add_node.out_arrays),
    )
    scope = next(sc for sc in trace.scope_calls if sc.address == "layers.3@0")
    emitted = emit_wrapper(trace, scope, [splice], "ArtLayer3")

    out = emit_artifact(tmp_path / "artifact", [kernel], [emitted], Report())
    assert (out / "kernels" / "k_art_add.metal").exists()
    assert (out / "patch" / "wrappers.py").exists()
    assert (out / "runtime" / "autotuner_runtime" / "kernels.py").exists()
    assert (out / "buffers").is_dir()

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
