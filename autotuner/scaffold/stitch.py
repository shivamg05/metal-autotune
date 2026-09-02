"""Stitched scaffolds: build the correct
starting kernel from the Metal source the installed mlx wheel ships, for ops
the naive lowerer cannot emit.

The wheel ships its MSL kernel tree under site-packages/mlx/include/mlx/
backend/metal/kernels/. The shipped [[kernel]] entry points cannot be
re-declared inside mx.fast.metal_kernel's body-function model, but the
algorithms live in METAL_FUNC device functions (qmv_fast_impl, qmv_impl) that
a kernel body may call directly, so the arithmetic that runs is the library's
own, verbatim. flatten_header inlines a root header's repo-relative includes
into one self-contained string, because metal_kernel resolves no nested
includes; it skips the utils.h prelude metal_kernel auto-prepends (re-inlining
it redefines its symbols) and ends with a newline (a header ending in a line
comment otherwise swallows the generated signature).

stitch_quantized_matmul covers mx.quantized_matmul(x, w, scales, biases,
transpose=True) for affine quantization. K and N are read at run time from
the injected inN_shape buffers, so one spec serves every (K, N) at a given
x rank; only group_size, bits, and the alignment variant are baked into the
kernel text. Verified bitwise against the library on this machine in
tests/test_stitch.py.

stitch_qmm_chain extends that to one quantized_matmul followed by row-local
elementwise ops (sigmoid, mul, add, ...): the qmv impl fills the 8 output
rows this threadgroup owns, and after a device-memory barrier the same
threadgroup applies the trailing chain to those rows in place.
"""

from __future__ import annotations

import functools
import hashlib
import pathlib
import re
from typing import Sequence

import mlx

from autotuner_runtime.kernels import KernelSpec

from ..trace.recorder import ArrayRef
from ..trace.types import TraceNode
# the naive lowerer's canonical elementwise tables and float-literal rule;
# reusing them keeps the op text (metal::precise:: included) identical
from .lower import _EW, _EW_NAMES, _flit
from .symshape import NoScaffold

Spec = tuple[Sequence[int], str]  # (shape, dtype name), the trace convention

_KERNELS = "mlx/backend/metal/kernels"
_PRELUDE = f"{_KERNELS}/utils.h"
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')

# MLX's own quantized build compiles the steel GEMM headers before
# quantized.h (its qmm sections use mlx::steel and elem_to_loc_broadcast).
_QMV_ROOTS = (f"{_KERNELS}/steel/gemm/gemm.h", f"{_KERNELS}/quantized.h")

_FLOATS = ("float16", "bfloat16", "float32")

# The impl reads K and N from the injected shape buffers (constant address
# space, so they bind to the const constant int& parameters). tid.x is the
# x row, tid.y the block of 8 output rows, matching the library's dispatch.
# {out} is the buffer the impl writes y into: out0 for the lone stitch, the
# region output that ships y (or out0 as scratch) for the fused chain.
_BODY = """\
{impl}<T, {group_size}, {bits}>(
    in1, in2, in3, in0, {out},
    in0_shape[in0_ndim - 1], in1_shape[0],
    threadgroup_position_in_grid,
    simdgroup_index_in_threadgroup,
    thread_index_in_simdgroup);
"""


def _include_root() -> pathlib.Path:
    return pathlib.Path(list(mlx.__path__)[0]) / "include"


@functools.lru_cache(maxsize=None)
def flatten_header(roots: tuple[str, ...]) -> str:
    """Inline the roots' repo-relative quoted includes, once each across all
    roots, into one metal_kernel-ready header. Skips the utils.h prelude set,
    strips #pragma once, keeps angle includes (Metal system headers), and
    terminates with a newline."""
    inc = _include_root()
    if not (inc / _PRELUDE).is_file():
        raise NoScaffold("stitch-no-shipped-msl", f"{inc / _PRELUDE} not found")
    seen: set[str] = set()
    _inline(_PRELUDE, inc, seen, [])  # prelude walk fills the skip set only
    lines: list[str] = []
    for root in roots:
        _inline(root, inc, seen, lines)
    return "\n".join(lines) + "\n"


def _inline(rel: str, inc: pathlib.Path, seen: set[str], out: list[str]) -> None:
    if rel in seen:
        return
    seen.add(rel)
    path = inc / rel
    if not path.is_file():
        raise NoScaffold("stitch-missing-header", rel)
    for line in path.read_text().splitlines():
        m = _INCLUDE_RE.match(line)
        if m:
            _inline(m.group(1), inc, seen, out)
        elif line.strip() != "#pragma once":
            out.append(line)


def stitch_quantized_matmul(
    x_spec: Spec,
    w_spec: Spec,
    scales_spec: Spec,
    biases_spec: Spec,
    group_size: int = 64,
    bits: int = 4,
) -> KernelSpec:
    """A KernelSpec reproducing mx.quantized_matmul(x, w, scales, biases,
    transpose=True, group_size, bits) by calling the wheel's own qmv impl.

    Specs are (shape, dtype name) pairs as the trace records them. The spec
    mirrors the library's dispatch rule: the fast variant when K divides into
    full blocks and N into blocks of 8 output rows, the guarded general
    variant otherwise, so on decode shapes (x rows M == 1) the output is
    bitwise identical to the library. Larger M stays correct via the grid but
    the library may pick a different kernel there, so agreement is only
    within accumulation-order rounding. Raises NoScaffold with a stable
    reason when the inputs are outside what this stitch supports.
    """
    (xs, xdt), (ws, wdt) = (tuple(x_spec[0]), x_spec[1]), (tuple(w_spec[0]), w_spec[1])
    (ss, sdt), (bs, bdt) = (tuple(scales_spec[0]), scales_spec[1]), (tuple(biases_spec[0]), biases_spec[1])
    if bits not in (4, 8):
        raise NoScaffold("stitch-bits", f"bits={bits}; only 4 and 8 are stitched")
    if xdt not in _FLOATS or sdt != xdt or bdt != xdt:
        raise NoScaffold("stitch-dtype", f"x={xdt} scales={sdt} biases={bdt}")
    if wdt != "uint32":
        raise NoScaffold("stitch-dtype", f"w={wdt}, want uint32 packing")
    if len(xs) < 1 or len(ws) != 2:
        raise NoScaffold("stitch-shape", f"x rank {len(xs)}, w rank {len(ws)}, want >=1 and 2")
    k, (n, kw) = xs[-1], ws
    pack_factor = 32 // bits
    if kw * pack_factor != k or group_size <= 0 or k % group_size:
        raise NoScaffold("stitch-shape", f"w {ws} does not pack K={k} at {bits} bits, group {group_size}")
    groups = (n, k // group_size)
    if ss != groups or bs != groups:
        raise NoScaffold("stitch-shape", f"scales {ss} biases {bs}, want {groups}")

    # library dispatch rule: qmv_fast needs whole blocks of 32 threads times
    # 2 packs, 8 output rows per threadgroup, and a scale step per thread
    vpt_fast = 2 * pack_factor
    fast = k % (32 * vpt_fast) == 0 and n % 8 == 0 and group_size % vpt_fast == 0
    if not fast and group_size % pack_factor:
        raise NoScaffold("stitch-group-size", f"group_size={group_size} not a multiple of {pack_factor}")

    # one threadgroup is 2 simdgroups of 32 covering 8 output rows; grid is
    # total threads: 32 per x row along x, 2 per 8-output-row block along y
    lead = [f"in0.shape[{i}]" for i in range(len(xs) - 1)]
    grid_x = " * ".join(lead + ["32"])
    name = f"stitch_affine_qmv_{'fast' if fast else 'gen'}_g{group_size}_b{bits}_r{len(xs)}"
    return KernelSpec(
        kernel_id=name,
        name=name,
        input_names=("in0", "in1", "in2", "in3"),  # x, w, scales, biases
        output_names=("out0",),
        source=_BODY.format(impl="qmv_fast_impl" if fast else "qmv_impl",
                            group_size=group_size, bits=bits, out="out0"),
        header=flatten_header(_QMV_ROOTS),
        grid=(grid_x, "ceil_div(in1.shape[0], 8) * 2", "1"),
        threadgroup=("32", "2", "1"),
        output_shapes=(tuple(lead) + ("in1.shape[0]",),),
        output_dtypes=(xdt,),
        template=(("T", "in0"),),
    )


# After the impl the barrier makes this threadgroup's 8 output rows visible
# to its own threads; 8 of the 64 then read the matmul value from {target},
# run the chain in registers, and store every region output for those rows.
# Safe only on the fast variant: qmv_fast_impl has no early returns (every
# thread reaches the barrier) and each block of 8 rows is written by exactly
# one threadgroup, so no other threadgroup touches these elements.
_CHAIN_EPILOGUE = """\
threadgroup_barrier(mem_flags::mem_device);
{{
    const int e_ = (int)thread_position_in_threadgroup.y * 32
        + (int)thread_position_in_threadgroup.x;
    if (e_ < 8) {{
        const int n_ = (int)threadgroup_position_in_grid.y * 8 + e_;
        const int F_ = (int)threadgroup_position_in_grid.x * in1_shape[0] + n_;
        const float y0_ = (float){target}[F_];
        {chain}
        {stores}
    }}
}}
"""


def _chain_operand(arg, node: TraceNode, env: dict, slot: dict,
                   out_shape: tuple, xdt: str, n: int) -> str:
    """The float expression for one elementwise operand: a chain value in a
    register, a python scalar, or a load from a region input that broadcasts
    or aligns to the matmul's output rows."""
    if isinstance(arg, ArrayRef):
        aid = node.in_arrays[arg.index]
        if aid in env:
            return env[aid]
        k = slot.get(aid)
        if k is None:
            raise NoScaffold("stitch-chain-operand",
                             f"{node.op} reads array {aid} produced outside the chain inputs")
        shape, dt = tuple(node.in_specs[arg.index][0]), node.in_specs[arg.index][1]
        if dt != xdt:
            raise NoScaffold("stitch-chain-dtype", f"{node.op} operand is {dt}, x is {xdt}")
        if all(d == 1 for d in shape):
            idx = "0"
        elif shape == out_shape:
            idx = "F_"
        elif shape[-1] == n and all(d == 1 for d in shape[:-1]):
            idx = "n_"
        else:
            raise NoScaffold("stitch-chain-broadcast",
                             f"{node.op} operand shape {shape} is not row-local to output {out_shape}")
        return f"((float)in{k}[{idx}])"
    if isinstance(arg, (bool, int, float)):
        return _flit(arg)
    raise NoScaffold("stitch-chain-args", f"{node.op} operand {arg!r}")


def stitch_qmm_chain(
    nodes: Sequence[TraceNode],
    input_ids: Sequence[int],
    output_ids: Sequence[int],
) -> KernelSpec:
    """A KernelSpec for a region that is one mx.quantized_matmul followed by
    row-local elementwise ops (the _EW table: sigmoid, mul, add, ...), fused
    into the stitched qmv kernel.

    The wheel's qmv impl writes the 8 output rows this threadgroup owns, then
    the epilogue applies the chain to those rows in the same launch. Each op
    computes in float and immediately rounds through the boundary dtype, as
    the library does per materialized op, so fp16 overflow to inf/nan happens
    at the same elements; the matmul bits are the library's own, and the
    chain is tolerance-level (library sigmoid differs from the precise::exp
    composition by 1 ulp, PLATFORM.md).
    Operands may be the running value, the matmul's own output, a python
    scalar, or a region input that is scalar-shaped, row-aligned (leading
    dims 1, last dim N), or exactly output-shaped. Every region output must
    be the matmul output or a chain value; a cut mid-chain (the matmul output
    consumed again after the region) ships each live value as its own kernel
    output. Only the fast qmv variant is fused: it has no early returns and
    each block of 8 output rows belongs to exactly one threadgroup, so the
    in-place epilogue cannot race. The guarded variant's edge threadgroup
    redoes rows of the previous block (N % 8 != 0), so chains off the fast
    alignment raise NoScaffold and the naive lowering takes the region.
    """
    if len(nodes) < 2:
        raise NoScaffold("stitch-chain-length", "no trailing elementwise ops")
    qmm = nodes[0]
    kw = qmm.scalar_args.get("kwargs", {})
    if qmm.op != "mx.quantized_matmul" or len(qmm.in_specs) != 4 \
            or len(qmm.scalar_args.get("args", ())) > 4 or len(qmm.out_arrays) != 1:
        raise NoScaffold("stitch-chain-head", f"{qmm.op} is not a plain 4-array quantized_matmul")
    if kw.get("transpose", True) is not True or kw.get("mode", "affine") != "affine":
        raise NoScaffold("stitch-chain-head", "only transposed affine quantized_matmul is stitched")
    if tuple(input_ids[:4]) != tuple(qmm.in_arrays):
        raise NoScaffold("stitch-chain-inputs", "qmm boundary inputs must fill slots 0..3")
    base = stitch_quantized_matmul(*qmm.in_specs[:4],
                                   group_size=kw.get("group_size", 64),
                                   bits=kw.get("bits", 4))
    if "qmv_fast_impl" not in base.source:
        raise NoScaffold("stitch-chain-variant",
                         "only the fast qmv variant is fused: the guarded one has "
                         "early returns and its edge threadgroup redoes rows")
    (xs, xdt) = tuple(qmm.in_specs[0][0]), qmm.in_specs[0][1]
    n = qmm.in_specs[1][0][0]
    out_shape = tuple(qmm.out_specs[0][0])
    if out_shape != xs[:-1] + (n,) or qmm.out_specs[0][1] != xdt:
        raise NoScaffold("stitch-chain-shape", f"qmm output spec {qmm.out_specs[0]}")
    if not output_ids:
        raise NoScaffold("stitch-chain-outputs", "region has no outputs")

    slot = {aid: k for k, aid in enumerate(input_ids)}
    env: dict[int, str] = {qmm.out_arrays[0]: "y0_"}
    lines: list[str] = []
    for i, node in enumerate(nodes[1:], start=1):
        if node.op not in _EW_NAMES:
            raise NoScaffold("stitch-chain-op", node.op)
        key, swapped = _EW_NAMES[node.op]
        fmt, arity, bool_out = _EW[key]
        args = list(node.scalar_args.get("args", ()))
        if node.scalar_args.get("kwargs", {}) or len(args) != arity:
            raise NoScaffold("stitch-chain-args", f"{node.op} takes {arity} plain args")
        if bool_out or len(node.out_arrays) != 1 \
                or tuple(node.out_specs[0][0]) != out_shape or node.out_specs[0][1] != xdt:
            raise NoScaffold("stitch-chain-shape", f"{node.op} -> {node.out_specs[0]}")
        if swapped:
            args.reverse()
        exprs = [_chain_operand(a, node, env, slot, out_shape, xdt, n) for a in args]
        result = fmt.format(a=exprs[0], b=exprs[1] if arity == 2 else "")
        # round-trip through T after every op: the library materializes the
        # boundary dtype per op, so fp16 overflow and rounding must land here
        lines.append(f"const float v{i}_ = (float)(T){result};")
        env[node.out_arrays[0]] = f"v{i}_"

    missing = [aid for aid in output_ids if aid not in env]
    if missing:
        raise NoScaffold("stitch-chain-outputs",
                         f"region outputs {missing} are not the matmul output or chain values")
    # the impl writes y into the output slot that ships it, or out0 as scratch
    # (the epilogue reads y0_ from the target before any store overwrites it)
    y_id = qmm.out_arrays[0]
    target = f"out{output_ids.index(y_id)}" if y_id in output_ids else "out0"
    stores = [f"out{p}[F_] = (T){env[aid]};"
              for p, aid in enumerate(output_ids) if aid != y_id]
    epilogue = _CHAIN_EPILOGUE.format(
        target=target, chain="\n        ".join(lines), stores="\n        ".join(stores))
    # the epilogue text keys the name: equal-length chains with different ops
    # or operand or output wiring must not share a kernel_id (the artifact
    # files by id)
    name = f"{base.name}_chain{len(nodes) - 1}_{hashlib.md5(epilogue.encode()).hexdigest()[:8]}"
    return KernelSpec(
        kernel_id=name,
        name=name,
        input_names=tuple(f"in{k}" for k in range(len(input_ids))),
        output_names=tuple(f"out{p}" for p in range(len(output_ids))),
        source=base.source.replace("in0, out0,", f"in0, {target},") + epilogue,
        header=base.header,
        grid=base.grid,
        threadgroup=base.threadgroup,
        output_shapes=base.output_shapes * len(output_ids),
        output_dtypes=(xdt,) * len(output_ids),
        template=base.template,
    )
