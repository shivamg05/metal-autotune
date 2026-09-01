"""Pinned platform facts from spike_08 (shipped MSL and the include flattener).

Behavior pins only, no timing. Each test names the design argument it protects,
so an mlx upgrade that breaks the scaffold stitcher (plan 5.9/5.10) fails loudly.
"""

import pathlib
import re

import pytest

import mlx
import mlx.core as mx

INC_ROOT = pathlib.Path(list(mlx.__path__)[0]) / "include"
KERNELS = INC_ROOT / "mlx/backend/metal/kernels"
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')


def run_kernel(name, source, header=""):
    """Compile and eval a 4-thread float32 kernel; return the output array."""
    k = mx.fast.metal_kernel(
        name=name, input_names=[], output_names=["out"], source=source, header=header
    )
    out = k(
        inputs=[],
        output_shapes=[(4,)],
        output_dtypes=[mx.float32],
        grid=(4, 1, 1),
        threadgroup=(4, 1, 1),
    )[0]
    mx.eval(out)
    return out


def flatten(rel, seen):
    """Inline repo-relative quoted includes once each; strip #pragma once."""
    path = INC_ROOT / rel
    seen.add(rel)
    lines = []
    for line in path.read_text().splitlines():
        m = INCLUDE_RE.match(line)
        if m:
            if m.group(1) not in seen:
                lines.append(flatten(m.group(1), seen))
        elif line.strip() != "#pragma once":
            lines.append(line)
    return "\n".join(lines)


def test_wheel_ships_msl_kernel_tree():
    """Design argument: scaffolds stitch from MLX's shipped MSL sources (plan
    5.9). If the wheel stops shipping the kernel tree, stitching loses its
    source and every scaffold silently degrades to naive lowering."""
    assert KERNELS.is_dir(), f"{KERNELS} missing from the installed wheel"
    for rel in ["steel/gemm/gemm.h", "sdpa_vector.h", "quantized.h", "utils.h"]:
        assert (KERNELS / rel).is_file(), f"{rel} missing under {KERNELS}"


def test_utils_prelude_auto_prepended():
    """Design argument: metal_kernel prepends utils.h to every kernel (plan
    5.10), so stitched source may use its symbols bare and the flattener must
    treat utils.h as already present."""
    out = run_kernel(
        "pin08_prelude",
        "uint i = thread_position_in_grid.x; out[i] = Limits<float>::max;",
    )
    assert [out[i].item() for i in range(4)] == [float("inf")] * 4


def test_absolute_include_resolves(tmp_path):
    """Design argument: the stitcher hands metal_kernel a header that includes
    the per-job flattened file by absolute path; if absolute includes stop
    resolving from disk, no stitched scaffold can compile."""
    hdr = tmp_path / "pin08_tiny.h"
    hdr.write_text("static inline float pin08_fn(float x) { return x * 3.0f + 1.0f; }\n")
    out = run_kernel(
        "pin08_abs",
        "uint i = thread_position_in_grid.x; out[i] = pin08_fn(float(i));",
        header=f'#include "{hdr}"\n',
    )
    assert [out[i].item() for i in range(4)] == [1.0, 4.0, 7.0, 10.0]


def test_shipped_header_nested_relative_includes_fail():
    """Design argument: the include flattener exists because shipped headers
    cannot be included directly; their nested repo-relative includes do not
    resolve. If this compile ever succeeds, mlx changed include resolution and
    the flattener is dead weight."""
    gemm = KERNELS / "steel/gemm/gemm.h"
    with pytest.raises(RuntimeError) as e:
        run_kernel(
            "pin08_nested",
            "uint i = thread_position_in_grid.x; out[i] = 0.0f;",
            header=f'#include "{gemm}"\n',
        )
    assert re.search(r"'mlx/[^']*' file not found", str(e.value))


def test_reinlined_utils_clashes_with_prelude():
    """Design argument: the flattener's skip set. Re-inlining utils.h next to
    the auto-prepended copy redefines its symbols, so the flattener must skip
    utils.h and its transitive includes. If this compiles, the prelude changed
    and the skip set needs rechecking."""
    flat_utils = flatten("mlx/backend/metal/kernels/utils.h", set()) + "\n"
    with pytest.raises(RuntimeError) as e:
        run_kernel(
            "pin08_utilsflat",
            "uint i = thread_position_in_grid.x; out[i] = 1.0f;",
            header=flat_utils,
        )
    assert "redefinition of" in str(e.value)
