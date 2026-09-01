"""Spike 08: include flattener feasibility (plan 5.9 stitching, M0).

Proves, on the installed mlx wheel:
  1. MSL kernel source ships under site-packages/mlx/include/mlx/backend/metal/kernels/.
  2. mx.fast.metal_kernel(header=...) resolves an absolute-path #include from disk.
  3. Shipped headers' nested repo-relative includes do NOT resolve when included absolutely.
  4. A one-time recursive include flattener yields a self-contained header that compiles.

Output: one `FACT <slug>: PASS|FAIL|INFO - <detail>` line per fact. Exit 0 if the
script ran to completion, even with FAILs.
"""

import pathlib
import re
import sys

import mlx
import mlx.core as mx

INC_ROOT = pathlib.Path(list(mlx.__path__)[0]) / "include"
KERNELS = INC_ROOT / "mlx/backend/metal/kernels"
OUT_DIR = pathlib.Path(__file__).resolve().parent / "out"
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')

facts = []


def fact(slug, status, detail):
    line = f"FACT {slug}: {status} - {detail}"
    facts.append(line)
    print(line, flush=True)


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


def first_line(exc):
    for ln in str(exc).splitlines():
        ln = ln.strip()
        if ln and "Unable to build" not in ln:
            return ln
    return str(exc)[:120]


# Fact 1: the wheel ships the MSL kernel tree.
def check_tree():
    key_files = [
        "steel/gemm/gemm.h",
        "steel/attn/attn.h",
        "sdpa_vector.h",
        "quantized.h",
        "reduction/reduce_row.h",
        "utils.h",
    ]
    if not KERNELS.is_dir():
        fact("shipped-msl-tree", "FAIL", f"{KERNELS} does not exist")
        return
    headers = sorted(KERNELS.rglob("*.h"))
    missing = [f for f in key_files if not (KERNELS / f).is_file()]
    total = sum(h.stat().st_size for h in headers)
    if missing:
        fact("shipped-msl-tree", "FAIL", f"tree exists but missing {missing}")
    else:
        fact(
            "shipped-msl-tree",
            "PASS",
            f"{len(headers)} .h files, {total} bytes under {KERNELS}; "
            "steel GEMM, sdpa, quantized, reduction all present",
        )
    sizes = ", ".join(f"{f}={ (KERNELS / f).stat().st_size }B" for f in key_files if (KERNELS / f).is_file())
    whole = sum(p.stat().st_size for p in INC_ROOT.rglob("*") if p.is_file())
    fact(
        "msl-tree-size",
        "INFO",
        f"kernels MSL subtree {total} bytes; whole include tree {whole} bytes "
        f"(plan 5.9 says ~4MB under kernels/, which matches the whole tree, not kernels/); {sizes}",
    )


# Fact 2: plan 5.10 says metal_kernel auto-prepends utils.h; the flattener must know.
def check_utils_prepended():
    try:
        out = run_kernel(
            "spike08_prelude",
            "uint i = thread_position_in_grid.x; out[i] = Limits<float>::max;",
        )
        vals = [out[i].item() for i in range(4)]
        if all(v == float("inf") for v in vals):
            fact(
                "utils-auto-prepended",
                "PASS",
                "empty-header kernel sees Limits<float>::max (=inf) from utils.h",
            )
        else:
            fact("utils-auto-prepended", "FAIL", f"unexpected values {vals}")
    except RuntimeError as e:
        fact("utils-auto-prepended", "FAIL", f"utils.h symbols not visible: {first_line(e)}")


# Fact 3: an absolute-path #include in header= resolves from disk.
def check_absolute_include():
    hdr = OUT_DIR / "spike08_tiny.h"
    hdr.write_text("static inline float spike08_fn(float x) { return x * 3.0f + 1.0f; }\n")
    try:
        out = run_kernel(
            "spike08_abs",
            "uint i = thread_position_in_grid.x; out[i] = spike08_fn(float(i));",
            header=f'#include "{hdr}"\n',
        )
        vals = [out[i].item() for i in range(4)]
        if vals == [1.0, 4.0, 7.0, 10.0]:
            fact(
                "absolute-include-resolves",
                "PASS",
                f'header \'#include "{hdr}"\' compiled and computed 3x+1 correctly',
            )
        else:
            fact("absolute-include-resolves", "FAIL", f"compiled but wrong values {vals}")
    except RuntimeError as e:
        fact("absolute-include-resolves", "FAIL", f"compile failed: {first_line(e)}")


# Fact 4: a shipped header's nested repo-relative includes do not resolve.
def check_nested_relative_fails():
    gemm = KERNELS / "steel/gemm/gemm.h"
    try:
        run_kernel(
            "spike08_nested",
            "uint i = thread_position_in_grid.x; out[i] = 0.0f;",
            header=f'#include "{gemm}"\n',
        )
        fact(
            "nested-relative-include-fails",
            "FAIL",
            "absolute include of steel/gemm/gemm.h compiled; nested includes resolved after all",
        )
    except RuntimeError as e:
        msg = str(e)
        m = re.search(r"fatal error: '([^']+)' file not found", msg)
        if m and m.group(1).startswith("mlx/"):
            fact(
                "nested-relative-include-fails",
                "PASS",
                f"compile failed as plan claims: '{m.group(1)}' file not found",
            )
        else:
            fact(
                "nested-relative-include-fails",
                "FAIL",
                f"failed for a different reason: {first_line(e)}",
            )


def flatten(rel, seen, inlined):
    """Inline repo-relative quoted includes once each; strip #pragma once."""
    if rel in seen:
        return ""
    path = INC_ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(rel)
    seen.add(rel)
    inlined.append(rel)
    lines = []
    for line in path.read_text().splitlines():
        m = INCLUDE_RE.match(line)
        if m:
            lines.append(flatten(m.group(1), seen, inlined))
        elif line.strip() == "#pragma once":
            continue
        else:
            lines.append(line)
    return "\n".join(lines)


PRELUDE = "mlx/backend/metal/kernels/utils.h"
GEMM = "mlx/backend/metal/kernels/steel/gemm/gemm.h"
PROBE_SRC = "uint i = thread_position_in_grid.x; out[i] = float(sizeof(mlx::steel::GEMMParams));"


# Fact 5: re-inlining utils.h collides with the auto-prepended copy, so the
# flattener must skip the prelude set. No prior plan claim; recorded as INFO.
def check_prelude_collision():
    flat_utils = flatten(PRELUDE, set(), []) + "\n"
    try:
        run_kernel(
            "spike08_utilsflat",
            "uint i = thread_position_in_grid.x; out[i] = 1.0f;",
            header=flat_utils,
        )
        fact(
            "flattened-utils-vs-prelude",
            "INFO",
            "flattened utils.h compiled alongside the auto-prepended copy; no skip set needed",
        )
    except RuntimeError as e:
        m = re.search(r"error: redefinition of '([^']+)'", str(e))
        detail = f"redefinition of '{m.group(1)}'" if m else first_line(e)
        fact(
            "flattened-utils-vs-prelude",
            "INFO",
            f"re-inlining utils.h clashes with the auto-prepended copy ({detail}); "
            "the flattener must skip utils.h and its transitive includes",
        )


# Fact 6: a header not ending in a newline lets a trailing line comment swallow
# the generated kernel signature. Measured trap, no prior claim; INFO.
def check_trailing_newline(flat_gemm):
    stripped = flat_gemm.rstrip("\n")
    if not stripped.endswith("// namespace mlx"):
        fact(
            "header-trailing-newline",
            "INFO",
            "flattened gemm.h no longer ends in a line comment; trap not testable here",
        )
        return
    try:
        run_kernel("spike08_nonewline", PROBE_SRC, header=stripped)
        fact(
            "header-trailing-newline",
            "INFO",
            "header without trailing newline compiled fine; no signature-swallow trap",
        )
    except RuntimeError as e:
        fact(
            "header-trailing-newline",
            "INFO",
            "header ending in a line comment with no trailing newline swallows the "
            f"generated kernel signature and fails to compile ({first_line(e)}); "
            "the flattener must emit a trailing newline",
        )


# Fact 7: the flattener is feasible; a flattened steel GEMM header compiles.
def check_flattener():
    prelude_seen = set()
    flatten(PRELUDE, prelude_seen, [])
    inlined = []
    flat = flatten(GEMM, set(prelude_seen), inlined) + "\n"
    (OUT_DIR / "spike08_gemm_flat.h").write_text(flat)
    check_trailing_newline(flat)
    try:
        out = run_kernel("spike08_gemmflat", PROBE_SRC, header=flat)
        vals = [out[i].item() for i in range(4)]
        if len(set(vals)) == 1 and vals[0] > 0:
            fact(
                "include-flattener-feasible",
                "PASS",
                f"flattened steel/gemm/gemm.h ({len(inlined)} files, {len(flat)} chars, "
                f"prelude skipped) compiles as header; sizeof(mlx::steel::GEMMParams)={int(vals[0])}",
            )
        else:
            fact("include-flattener-feasible", "FAIL", f"compiled but bad output {vals}")
    except RuntimeError as e:
        fact("include-flattener-feasible", "FAIL", f"flattened header did not compile: {first_line(e)}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    fact("mlx-version", "INFO", f"mlx {mx.__version__} at {list(mlx.__path__)[0]}")
    check_tree()
    check_utils_prepended()
    check_absolute_include()
    check_nested_relative_fails()
    check_prelude_collision()
    check_flattener()
    return 0


if __name__ == "__main__":
    sys.exit(main())
