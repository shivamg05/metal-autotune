"""Naive lowering: one correct Metal kernel per region.

lower_naive turns a region stretch into a KernelSpec that runs the recorded
ops back to back, correctness first. Each compute node becomes a stage that
writes a device buffer: region outputs first (out0..), then scratch buffers
declared as extra kernel outputs (tmp0..). View nodes fold into their
consumers' index arithmetic as stride algebra; a reshape of a non-contiguous
view materializes through a copy stage, exactly as the library does.

Two launch modes, chosen by analysis. When every stage has the same row count
and every read of a stage-written buffer stays inside its own row, one
threadgroup owns each row and runs all stages for it with device-memory
barriers in between (rms_norm and row reductions use a cooperative
threadgroup-memory tree, fixed order, so results are deterministic).
Otherwise a single threadgroup runs the whole region, still correct, just
serial across rows. The ladder, not this module, decides whether it is fast.

Launch sizes and output shapes are launch-grammar expressions over the
boundary inputs; a dim bakes as a literal only when it is constant across
every provided instance. fp16/bf16 stages accumulate in float (a legal
internal precision); transcendentals use metal::precise:: so they match
library bits.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Sequence

from autotuner_runtime.kernels import KernelSpec

from ..regions.types import Stretch
from ..trace.recorder import ArrayRef
from ..trace.types import Trace, TraceNode
from .symshape import (
    Dim,
    NoScaffold,
    View,
    accessor,
    broadcast_shapes,
    contiguous,
    dims_equal,
    drop_axes,
    insert_axis,
    is_contiguous,
    lit,
    permute,
    prod_dims,
    shapes_equal,
)

TGX = 128  # threads per threadgroup; the reduction tree assumes a power of two
RB = 8     # rows per threadgroup when a rank-2 matmul is present (register blocking)

_MSL = {"float32": "float", "float16": "half", "bfloat16": "bfloat16_t", "bool": "bool"}
_FLOATS = ("float32", "float16", "bfloat16")

# canonical elementwise ops: key -> (C format over {a}/{b}, arity, bool output)
_EW = {
    "add": ("({a} + {b})", 2, False),
    "sub": ("({a} - {b})", 2, False),
    "mul": ("({a} * {b})", 2, False),
    "div": ("({a} / {b})", 2, False),
    # mx.maximum/minimum propagate NaN (numpy semantics); metal::max does
    # not, so a NaN operand routes through addition, which does propagate
    "maximum": ("(((({a}) != ({a})) || (({b}) != ({b}))) ? (({a}) + ({b})) : metal::max({a}, {b}))", 2, False),
    "minimum": ("(((({a}) != ({a})) || (({b}) != ({b}))) ? (({a}) + ({b})) : metal::min({a}, {b}))", 2, False),
    "neg": ("(-{a})", 1, False),
    "abs": ("metal::abs({a})", 1, False),
    "square": ("({a} * {a})", 1, False),
    "exp": ("metal::precise::exp({a})", 1, False),
    "tanh": ("metal::precise::tanh({a})", 1, False),
    "sigmoid": ("(1.0f / (1.0f + metal::precise::exp(-{a})))", 1, False),
    "sqrt": ("metal::precise::sqrt({a})", 1, False),
    "rsqrt": ("metal::precise::rsqrt({a})", 1, False),
    "sin": ("metal::precise::sin({a})", 1, False),
    "cos": ("metal::precise::cos({a})", 1, False),
    "lt": ("({a} < {b})", 2, True),
    "le": ("({a} <= {b})", 2, True),
    "gt": ("({a} > {b})", 2, True),
    "ge": ("({a} >= {b})", 2, True),
    "eq": ("({a} == {b})", 2, True),
    "ne": ("({a} != {b})", 2, True),
}

# trace op name -> (canonical key, operands reversed)
_EW_NAMES = {
    "mx.add": ("add", False), "array.__add__": ("add", False), "array.__radd__": ("add", True),
    "mx.subtract": ("sub", False), "array.__sub__": ("sub", False), "array.__rsub__": ("sub", True),
    "mx.multiply": ("mul", False), "array.__mul__": ("mul", False), "array.__rmul__": ("mul", True),
    "mx.divide": ("div", False), "array.__truediv__": ("div", False), "array.__rtruediv__": ("div", True),
    "mx.maximum": ("maximum", False), "mx.minimum": ("minimum", False),
    "mx.negative": ("neg", False), "array.__neg__": ("neg", False),
    "mx.abs": ("abs", False), "array.__abs__": ("abs", False), "array.abs": ("abs", False),
    "mx.square": ("square", False), "array.square": ("square", False),
    "mx.exp": ("exp", False), "array.exp": ("exp", False),
    "mx.tanh": ("tanh", False), "mx.sigmoid": ("sigmoid", False),
    "mx.sqrt": ("sqrt", False), "array.sqrt": ("sqrt", False),
    "mx.rsqrt": ("rsqrt", False), "array.rsqrt": ("rsqrt", False),
    "mx.sin": ("sin", False), "mx.cos": ("cos", False),
    "mx.less": ("lt", False), "array.__lt__": ("lt", False),
    "mx.less_equal": ("le", False), "array.__le__": ("le", False),
    "mx.greater": ("gt", False), "array.__gt__": ("gt", False),
    "mx.greater_equal": ("ge", False), "array.__ge__": ("ge", False),
    "mx.equal": ("eq", False), "array.__eq__": ("eq", False),
    "mx.not_equal": ("ne", False), "array.__ne__": ("ne", False),
}

# reduce op name -> (init, combine format over {a}/{x}, divide by row length)
_REDUCE = {
    "sum": ("0.0f", "({a} + {x})", False),
    "mean": ("0.0f", "({a} + {x})", True),
    "max": ("-INFINITY", "(((({a}) != ({a})) || (({x}) != ({x}))) ? (({a}) + ({x})) : metal::max({a}, {x}))", False),
    "min": ("INFINITY", "(((({a}) != ({a})) || (({x}) != ({x}))) ? (({a}) + ({x})) : metal::min({a}, {x}))", False),
}
_REDUCE_NAMES = {
    "mx.sum": "sum", "array.sum": "sum", "mx.mean": "mean", "array.mean": "mean",
    "mx.max": "max", "array.max": "max", "mx.min": "min", "array.min": "min",
}

_MATMUL_NAMES = frozenset({"mx.matmul", "array.__matmul__"})
_RMS_NAMES = frozenset({"mx.fast.rms_norm"})
_QMM_NAMES = frozenset({"mx.quantized_matmul"})
_ROPE_NAMES = frozenset({"mx.fast.rope"})
_CAST_NAMES = frozenset({"array.astype", "mx.astype"})
_VIEW_NAMES = frozenset({
    "array.reshape", "mx.reshape", "array.transpose", "mx.transpose", "array.T",
    "array.squeeze", "mx.squeeze", "mx.expand_dims",
})


@dataclass
class _Buffer:
    name: str            # in0.. / out0.. / tmp0..
    kind: str            # input | output | interm
    dtype: str           # recorded dtype name
    shape: tuple[Dim, ...]


@dataclass(frozen=True)
class _Value:
    buf: _Buffer
    view: View


@dataclass
class _Stage:
    kind: str                       # ew | copy | matmul | rms | reduce | qmm
    op: str                         # recorded op name, for messages
    out: _Buffer | None
    srcs: list = field(default_factory=list)  # _Value or python scalars, arg order
    fmt: str = ""
    bool_out: bool = False
    reduce_op: str = ""
    eps: float = 0.0
    qmm_bits: int = 0
    qmm_group: int = 0
    rope_args: tuple = ()           # (dims, traditional, base, scale, offset)


def stretch_input_shapes(trace: Trace, stretch: Stretch) -> tuple[tuple[int, ...], ...]:
    """The boundary input shapes in stretch.input_ids order, from the recorded
    specs. One traced size's stretch (a locate_span result included) is one
    lowering instance."""
    specs = _first_specs(trace.nodes[stretch.start_seq:stretch.end_seq + 1])
    return tuple(specs[aid][0] for aid in stretch.input_ids)


def _first_specs(nodes: Sequence[TraceNode]) -> dict[int, tuple]:
    specs: dict[int, tuple] = {}
    for node in nodes:
        for aid, spec in zip(node.in_arrays, node.in_specs):
            specs.setdefault(aid, spec)
        for aid, spec in zip(node.out_arrays, node.out_specs):
            specs.setdefault(aid, spec)
    return specs


class _Lowering:
    def __init__(self, trace: Trace, stretch: Stretch, instances):
        self.nodes = trace.nodes[stretch.start_seq:stretch.end_seq + 1]
        self.stretch = stretch
        self.n_inst = 1 + len(instances)
        self.specs = _first_specs(self.nodes)
        self.env: dict[int, _Value] = {}
        self.stages: list[_Stage] = []
        self.in_bufs: list[_Buffer] = []
        self.out_bufs: dict[int, _Buffer] = {}   # output position -> buffer
        self.tmp_bufs: list[_Buffer] = []
        self.out_ids = list(stretch.output_ids) or [self.nodes[-1].out_arrays[0]]
        self.out_pos = {aid: p for p, aid in enumerate(self.out_ids)}
        self._bind_inputs(instances)

    # -- inputs and buffers --------------------------------------------------

    def _bind_inputs(self, instances) -> None:
        in_ids = self.stretch.input_ids
        primary = [self.specs[aid][0] for aid in in_ids]
        shape_sets = [primary]
        for inst in instances:
            if len(inst) != len(in_ids):
                raise NoScaffold("bad-instance", f"{len(inst)} shapes for {len(in_ids)} inputs")
            shape_sets.append([tuple(s) for s in inst])
        for k, aid in enumerate(in_ids):
            rank = len(primary[k])
            if any(len(ss[k]) != rank for ss in shape_sets):
                raise NoScaffold("rank-varies", f"input {k} rank differs across instances")
            dims = []
            for j in range(rank):
                vals = tuple(ss[k][j] for ss in shape_sets)
                dims.append(lit(vals[0], self.n_inst) if len(set(vals)) == 1
                            else accessor(k, j, vals))
            buf = _Buffer(f"in{k}", "input", self.specs[aid][1], tuple(dims))
            self.in_bufs.append(buf)
            self.env[aid] = _Value(buf, contiguous(buf.shape, self.n_inst))

    def _stage_buffer(self, aid: int, dtype: str, shape: tuple[Dim, ...]) -> _Buffer:
        if dtype not in _MSL:
            raise NoScaffold("dtype-not-lowered", dtype)
        p = self.out_pos.get(aid)
        if p is not None:
            buf = _Buffer(f"out{p}", "output", dtype, shape)
            self.out_bufs[p] = buf
        else:
            buf = _Buffer(f"tmp{len(self.tmp_bufs)}", "interm", dtype, shape)
            self.tmp_bufs.append(buf)
        return buf

    def materialize(self, v: _Value, op: str) -> _Value:
        """Copy a strided view into a fresh contiguous scratch buffer."""
        if v.buf.dtype not in _MSL:
            raise NoScaffold("dtype-not-lowered", v.buf.dtype)
        buf = _Buffer(f"tmp{len(self.tmp_bufs)}", "interm", v.buf.dtype, v.view.shape)
        self.tmp_bufs.append(buf)
        self.stages.append(_Stage(kind="copy", op=op, out=buf, srcs=[v]))
        return _Value(buf, contiguous(buf.shape, self.n_inst))

    # -- the node walk -------------------------------------------------------

    def run(self) -> KernelSpec:
        for node in self.nodes:
            if len(node.out_arrays) != 1:
                raise NoScaffold("multi-output-op", node.op)
            if node.op in _CAST_NAMES:
                value = self._apply_cast(node)
            elif node.op in _VIEW_NAMES:
                value = self._apply_view(node)
            else:
                value = self._build_stage(node)
            got = tuple(d.values[0] for d in value.view.shape)
            if got != node.out_specs[0][0]:
                raise NoScaffold(
                    "shape-inference-mismatch",
                    f"{node.op} inferred {got}, recorded {node.out_specs[0][0]}",
                )
            self.env[node.out_arrays[0]] = value
        self._materialize_outputs()
        return self._assemble()

    def _materialize_outputs(self) -> None:
        """Every region output must fill its own contiguous out buffer; an
        output still living as a view (or an alias) gets a copy stage."""
        for aid in self.out_ids:
            p = self.out_pos[aid]
            v = self.env.get(aid)
            if v is None:
                raise NoScaffold("output-not-produced", f"array {aid}")
            direct = (
                p in self.out_bufs
                and v.buf is self.out_bufs[p]
                and is_contiguous(v.view, self.n_inst)
                and shapes_equal(v.view.shape, v.buf.shape)
            )
            if direct:
                continue
            if v.buf.dtype not in _MSL:
                raise NoScaffold("dtype-not-lowered", v.buf.dtype)
            buf = _Buffer(f"out{p}", "output", v.buf.dtype, v.view.shape)
            self.out_bufs[p] = buf
            self.stages.append(_Stage(kind="copy", op="output-copy", out=buf, srcs=[v]))

    def _operands(self, node: TraceNode):
        def resolve(obj):
            if isinstance(obj, ArrayRef):
                return self.env[node.in_arrays[obj.index]]
            return obj
        args = [resolve(a) for a in node.scalar_args["args"]]
        kwargs = {k: resolve(v) for k, v in node.scalar_args["kwargs"].items()}
        return args, kwargs

    # -- stage builders ------------------------------------------------------

    def _build_stage(self, node: TraceNode) -> _Value:
        if node.op in _EW_NAMES:
            stage, shape = self._ew_stage(node)
        elif node.op in _MATMUL_NAMES:
            stage, shape = self._matmul_stage(node)
        elif node.op in _RMS_NAMES:
            stage, shape = self._rms_stage(node)
        elif node.op in _QMM_NAMES:
            stage, shape = self._qmm_stage(node)
        elif node.op in _ROPE_NAMES:
            stage, shape = self._rope_stage(node)
        elif node.op in _REDUCE_NAMES:
            stage, shape = self._reduce_stage(node)
        else:
            raise NoScaffold("op-not-lowered", node.op)
        stage.out = self._stage_buffer(node.out_arrays[0], node.out_specs[0][1], shape)
        self.stages.append(stage)
        return _Value(stage.out, contiguous(shape, self.n_inst))

    def _float_values(self, node: TraceNode, operands) -> list[_Value]:
        values = [o for o in operands if isinstance(o, _Value)]
        if not values:
            raise NoScaffold("op-args-not-lowered", f"{node.op} has no array operand")
        dtypes = {v.buf.dtype for v in values}
        if not dtypes <= set(_FLOATS):
            raise NoScaffold("dtype-not-lowered", f"{node.op} on {sorted(dtypes)}")
        if len(dtypes) != 1:
            raise NoScaffold("dtype-not-lowered", f"{node.op} mixes {sorted(dtypes)}")
        return values

    def _ew_stage(self, node: TraceNode):
        key, swapped = _EW_NAMES[node.op]
        fmt, arity, bool_out = _EW[key]
        args, kwargs = self._operands(node)
        if kwargs or len(args) != arity:
            raise NoScaffold("op-args-not-lowered", f"{node.op} takes {arity} args")
        if swapped:
            args = args[::-1]
        for a in args:
            if not isinstance(a, (_Value, int, float, bool)):
                raise NoScaffold("op-args-not-lowered", f"{node.op} arg {a!r}")
        values = self._float_values(node, args)
        shape = values[0].view.shape
        for v in values[1:]:
            shape = broadcast_shapes(shape, v.view.shape)
        want = "bool" if bool_out else values[0].buf.dtype
        if node.out_specs[0][1] != want:
            raise NoScaffold("dtype-not-lowered", f"{node.op} promotes to {node.out_specs[0][1]}")
        return _Stage(kind="ew", op=node.op, out=None, srcs=args, fmt=fmt,
                      bool_out=bool_out), shape

    def _matmul_stage(self, node: TraceNode):
        args, kwargs = self._operands(node)
        if kwargs or len(args) != 2 or not all(isinstance(a, _Value) for a in args):
            raise NoScaffold("op-args-not-lowered", node.op)
        a, b = args
        self._float_values(node, args)
        if len(a.view.shape) < 2 or len(b.view.shape) < 2:
            raise NoScaffold("matmul-rank", "1-d operands are not lowered")
        if not dims_equal(a.view.shape[-1], b.view.shape[-2]):
            raise NoScaffold("matmul-shape", "inner dims disagree")
        batch = broadcast_shapes(a.view.shape[:-2], b.view.shape[:-2])
        shape = batch + (a.view.shape[-2], b.view.shape[-1])
        return _Stage(kind="matmul", op=node.op, out=None, srcs=[a, b]), shape

    def _rms_stage(self, node: TraceNode):
        args, kwargs = self._operands(node)
        x = args[0] if args else None
        w = args[1] if len(args) > 1 else kwargs.pop("weight", None)
        eps = kwargs.pop("eps", args[2] if len(args) > 2 else None)
        if kwargs or not isinstance(eps, (int, float)) or \
                not (isinstance(x, _Value) and isinstance(w, _Value)):
            raise NoScaffold("op-args-not-lowered", node.op)
        self._float_values(node, [x, w])
        if len(w.view.shape) != 1 or not dims_equal(w.view.shape[0], x.view.shape[-1]):
            raise NoScaffold("op-args-not-lowered", "rms_norm weight shape")
        return _Stage(kind="rms", op=node.op, out=None, srcs=[x, w],
                      eps=float(eps)), x.view.shape

    def _qmm_stage(self, node: TraceNode):
        """mx.quantized_matmul, affine mode, transposed weights: the layout
        every mlx quantized checkpoint uses. w packs 32/bits values per uint32
        along k; scales and biases are per group of group_size k values."""
        args, kwargs = self._operands(node)
        scales = kwargs.pop("scales", args[2] if len(args) > 2 else None)
        biases = kwargs.pop("biases", args[3] if len(args) > 3 else None)
        transpose = kwargs.pop("transpose", True)
        group_size = kwargs.pop("group_size", 64)
        bits = kwargs.pop("bits", 4)
        mode = kwargs.pop("mode", "affine")
        if kwargs or len(args) < 2:
            raise NoScaffold("op-args-not-lowered", f"{node.op} kwargs {sorted(kwargs)}")
        x, wq = args[0], args[1]
        if transpose is not True or mode != "affine" or bits not in (4, 8) \
                or not isinstance(group_size, int) or group_size < 1:
            raise NoScaffold("op-args-not-lowered",
                             f"{node.op} transpose={transpose} mode={mode} bits={bits}")
        if not all(isinstance(v, _Value) for v in (x, wq, scales, biases)):
            raise NoScaffold("op-args-not-lowered", node.op)
        self._float_values(node, [x, scales, biases])
        if wq.buf.dtype != "uint32":
            raise NoScaffold("dtype-not-lowered", f"{node.op} weights are {wq.buf.dtype}")
        for name, v in (("weight", wq), ("scales", scales), ("biases", biases)):
            if v.buf.kind != "input" or not _flat(v, v.buf.shape, self.n_inst) \
                    or not all(d.is_literal for d in v.view.shape):
                raise NoScaffold("op-args-not-lowered",
                                 f"{node.op} {name} is not a plain constant-shape input")
        if len(wq.view.shape) != 2 or len(scales.view.shape) != 2 \
                or not shapes_equal(scales.view.shape, biases.view.shape):
            raise NoScaffold("op-args-not-lowered", f"{node.op} weight ranks")
        per = 32 // bits
        if group_size % per:
            raise NoScaffold("op-args-not-lowered", f"{node.op} group_size {group_size} vs packing {per}")
        n_out = wq.view.shape[0].values[0]
        k = wq.view.shape[1].values[0] * per
        groups = scales.view.shape[1].values[0]
        if groups * group_size != k or scales.view.shape[0].values[0] != n_out:
            raise NoScaffold("op-args-not-lowered", f"{node.op} group layout")
        if any(v != k for v in x.view.shape[-1].values):
            raise NoScaffold("matmul-shape", "inner dims disagree")
        if node.out_specs[0][1] != x.buf.dtype:
            raise NoScaffold("dtype-not-lowered", f"{node.op} promotes to {node.out_specs[0][1]}")
        shape = x.view.shape[:-1] + (lit(n_out, self.n_inst),)
        return _Stage(kind="qmm", op=node.op, out=None, srcs=[x, wq, scales, biases],
                      qmm_bits=bits, qmm_group=group_size), shape

    def _rope_stage(self, node: TraceNode):
        """mx.fast.rope: rotate the leading `dims` of the last axis by a
        position-dependent angle; position is the index along axis -2 plus a
        constant offset (verified against the library for both variants)."""
        args, kwargs = self._operands(node)
        x = args[0] if args else None
        dims = kwargs.pop("dims", args[1] if len(args) > 1 else None)
        traditional = kwargs.pop("traditional", None)
        base = kwargs.pop("base", None)
        scale = kwargs.pop("scale", 1.0)
        offset = kwargs.pop("offset", 0)
        freqs = kwargs.pop("freqs", None)
        if kwargs or freqs is not None:
            raise NoScaffold("op-args-not-lowered",
                             f"{node.op} kwargs {sorted(kwargs) + (['freqs'] if freqs is not None else [])}")
        if not isinstance(x, _Value) or not isinstance(dims, int) \
                or not isinstance(traditional, bool) \
                or not isinstance(base, (int, float)) or not isinstance(scale, (int, float)) \
                or not isinstance(offset, int):
            raise NoScaffold("op-args-not-lowered", f"{node.op} arguments")
        self._float_values(node, [x])
        if len(x.view.shape) < 2:
            raise NoScaffold("op-args-not-lowered", f"{node.op} needs a position axis")
        last = x.view.shape[-1]
        if dims < 2 or dims % 2 or any(dims > v for v in last.values):
            raise NoScaffold("op-args-not-lowered", f"{node.op} dims {dims} vs last axis")
        if node.out_specs[0][1] != x.buf.dtype:
            raise NoScaffold("dtype-not-lowered", f"{node.op} promotes to {node.out_specs[0][1]}")
        return _Stage(kind="rope", op=node.op, out=None, srcs=[x],
                      rope_args=(dims, traditional, float(base), float(scale), offset)), x.view.shape

    def _reduce_stage(self, node: TraceNode):
        args, kwargs = self._operands(node)
        x = args[0] if args else None
        if not isinstance(x, _Value):
            raise NoScaffold("op-args-not-lowered", node.op)
        self._float_values(node, [x])
        axis = kwargs.pop("axis", args[1] if len(args) > 1 else None)
        keepdims = bool(kwargs.pop("keepdims", args[2] if len(args) > 2 else False))
        if kwargs:
            raise NoScaffold("op-args-not-lowered", f"{node.op} kwargs {sorted(kwargs)}")
        rank = len(x.view.shape)
        if isinstance(axis, (tuple, list)) and len(axis) == 1:
            axis = axis[0]
        if axis is None and rank == 1:
            axis = 0
        if not isinstance(axis, int) or axis % rank != rank - 1:
            raise NoScaffold("reduction-axis-not-last", f"{node.op} axis {axis}")
        shape = x.view.shape[:-1] + ((lit(1, self.n_inst),) if keepdims else ())
        return _Stage(kind="reduce", op=node.op, out=None, srcs=[x],
                      reduce_op=_REDUCE_NAMES[node.op]), shape

    # -- cast ----------------------------------------------------------------

    def _apply_cast(self, node: TraceNode) -> _Value:
        """astype: a same-shape dtype change. A no-op cast folds away; a real
        one is a copy stage whose store casts to the target dtype (a cast the
        recorded ops already contain, so the frozen-dtype law is not broken)."""
        args, _ = self._operands(node)
        if not args or not isinstance(args[0], _Value):
            raise NoScaffold("op-args-not-lowered", node.op)
        v = args[0]
        target = node.out_specs[0][1]
        if target == v.buf.dtype:
            return v
        if target not in _MSL:
            raise NoScaffold("dtype-not-lowered", f"{node.op} to {target}")
        buf = self._stage_buffer(node.out_arrays[0], target, v.view.shape)
        self.stages.append(_Stage(kind="copy", op=node.op, out=buf, srcs=[v]))
        return _Value(buf, contiguous(v.view.shape, self.n_inst))

    # -- view folding --------------------------------------------------------

    def _apply_view(self, node: TraceNode) -> _Value:
        args, kwargs = self._operands(node)
        if not args or not isinstance(args[0], _Value):
            raise NoScaffold("op-args-not-lowered", node.op)
        v, rest, op = args[0], args[1:], node.op
        if op in ("array.reshape", "mx.reshape"):
            return self._view_reshape(node, v, rest, kwargs)
        if op in ("array.transpose", "mx.transpose"):
            axes = self._int_list(rest, kwargs.pop("axes", None))
            if kwargs:
                raise NoScaffold("op-args-not-lowered", f"{op} kwargs {sorted(kwargs)}")
            rank = len(v.view.shape)
            axes = [a % rank for a in axes] if axes else list(range(rank))[::-1]
            if sorted(axes) != list(range(rank)):
                raise NoScaffold("op-args-not-lowered", f"{op} axes {axes}")
            return _Value(v.buf, permute(v.view, axes))
        if op == "array.T":
            return _Value(v.buf, permute(v.view, list(range(len(v.view.shape)))[::-1]))
        if op in ("array.squeeze", "mx.squeeze"):
            axes = self._int_list(rest, kwargs.pop("axis", None))
            if kwargs:
                raise NoScaffold("op-args-not-lowered", f"{op} kwargs {sorted(kwargs)}")
            rank = len(v.view.shape)
            drop = {a % rank for a in axes} if axes else {
                i for i, d in enumerate(v.view.shape) if d.is_one
            }
            for i in drop:
                if not v.view.shape[i].is_one:
                    raise NoScaffold("op-args-not-lowered", f"squeeze of non-unit axis {i}")
            return _Value(v.buf, drop_axes(v.view, drop))
        if op == "mx.expand_dims":
            axes = self._int_list(rest, kwargs.pop("axis", None))
            if kwargs or not axes:
                raise NoScaffold("op-args-not-lowered", op)
            view = v.view
            rank_out = len(view.shape) + len(axes)
            for a in sorted(x % rank_out for x in axes):
                view = insert_axis(view, a, self.n_inst)
            return _Value(v.buf, view)
        raise NoScaffold("view-not-lowered", op)

    def _int_list(self, rest, kw) -> list[int]:
        items = list(rest) if kw is None else [kw]
        flat: list[int] = []
        for it in items:
            if isinstance(it, (tuple, list)):
                flat.extend(it)
            elif it is not None:
                flat.append(it)
        if not all(isinstance(x, int) for x in flat):
            raise NoScaffold("op-args-not-lowered", f"non-int axes {flat}")
        return flat

    def _view_reshape(self, node: TraceNode, v: _Value, rest, kwargs) -> _Value:
        target = self._int_list(rest, kwargs.pop("shape", None))
        if kwargs:
            raise NoScaffold("op-args-not-lowered", f"reshape kwargs {sorted(kwargs)}")
        if not is_contiguous(v.view, self.n_inst):
            v = self.materialize(v, node.op)  # the library copies here too
        numel = prod_dims(v.view.shape, self.n_inst)
        known, minus = 1, None
        for i, t in enumerate(target):
            if t == -1:
                if minus is not None:
                    raise NoScaffold("op-args-not-lowered", "reshape with two -1 dims")
                minus = i
            else:
                known *= t
        dims = [lit(t, self.n_inst) for t in target if t != -1]
        if minus is None:
            if any(nv != known for nv in numel.values):
                raise NoScaffold(
                    "reshape-baked",
                    f"target {target} fits only one of the instance sizes {numel.values}",
                )
        else:
            if known == 0 or any(nv % known for nv in numel.values):
                raise NoScaffold("reshape-baked", f"target {target} does not divide {numel.values}")
            quots = tuple(nv // known for nv in numel.values)
            if len(set(quots)) == 1:
                inferred = lit(quots[0], self.n_inst)
            else:
                inferred = Dim(quots, f"({numel.grammar} // {known})",
                               f"(({numel.body}) / {known})")
            dims.insert(minus, inferred)
        shape = tuple(dims)
        return _Value(v.buf, contiguous(shape, self.n_inst))

    # -- launch mode analysis ------------------------------------------------

    def _stage_rows(self, st: _Stage) -> Dim:
        if st.kind in ("rms", "reduce"):
            return prod_dims(st.srcs[0].view.shape[:-1], self.n_inst)
        return prod_dims(st.out.shape[:-1], self.n_inst)

    def _row_local(self, st: _Stage, rows: Dim) -> bool:
        """True when every read of a stage-written buffer stays inside the row
        this threadgroup owns. Region inputs are read-only, so they never
        constrain the partition."""
        def own_row(v: _Value, want_shape) -> bool:
            if v.buf.kind == "input":
                return True
            return (is_contiguous(v.view, self.n_inst)
                    and shapes_equal(v.view.shape, v.buf.shape)
                    and (want_shape is None or shapes_equal(v.view.shape, want_shape))
                    and dims_equal(prod_dims(v.view.shape[:-1], self.n_inst), rows))
        if st.kind in ("ew", "copy"):
            return all(own_row(s, st.out.shape) for s in st.srcs if isinstance(s, _Value))
        if st.kind == "matmul":
            a, b = st.srcs
            return b.buf.kind == "input" and own_row(a, None)
        if st.kind == "rms":
            x, w = st.srcs
            return w.buf.kind == "input" and own_row(x, None)
        if st.kind == "qmm":
            # weight, scales, biases are plain inputs by construction
            return own_row(st.srcs[0], None)
        if st.kind == "rope":
            # both rotation partners live on the same row (same axis -2 index)
            return own_row(st.srcs[0], None)
        if st.kind == "reduce":
            return own_row(st.srcs[0], None)
        return False

    def _choose_mode(self) -> tuple[str, Dim]:
        rows = self._stage_rows(self.stages[0])
        multi = all(dims_equal(self._stage_rows(st), rows) for st in self.stages) \
            and all(self._row_local(st, rows) for st in self.stages)
        return ("multi" if multi else "single"), rows

    # -- assembly ------------------------------------------------------------

    def _assemble(self) -> KernelSpec:
        mode, rows = self._choose_mode()
        # a rank-2 matmul streams its whole B operand per row, so it gets RB
        # rows per threadgroup and B is read once per block, not once per row
        rb = RB if mode == "multi" and any(
            st.kind == "matmul" and len(st.out.shape) == 2 for st in self.stages
        ) else 1
        if mode != "multi":
            grid_y = "1"
        elif rb == 1:
            grid_y = rows.grammar
        else:
            grid_y = f"ceil_div({rows.grammar}, {rb})"
        outputs = [self.out_bufs[p] for p in range(len(self.out_ids))] + self.tmp_bufs
        msl = dict(_MSL)
        floats = {b.dtype for b in self.in_bufs + outputs if b.dtype in _FLOATS}
        template: tuple = ()
        if len(floats) == 1:
            f = next(iter(floats))
            for k, b in enumerate(self.in_bufs):
                if b.dtype == f:
                    template = (("T", f"in{k}"),)
                    msl[f] = "T"
                    break
        source = _emit_body(self.stages, _Ctx(mode, msl, self.n_inst, rb, rows.body))
        name = _kernel_name(self.nodes, self.stretch)
        return KernelSpec(
            kernel_id=name,
            name=name,
            input_names=tuple(b.name for b in self.in_bufs),
            output_names=tuple(b.name for b in outputs),
            source=source,
            grid=(str(TGX), grid_y, "1"),
            threadgroup=(str(TGX), "1", "1"),
            output_shapes=tuple(tuple(d.grammar for d in b.shape) for b in outputs),
            output_dtypes=tuple(b.dtype for b in outputs),
            template=template,
        )


def _kernel_name(nodes, stretch: Stretch) -> str:
    shorts = [n.op.rsplit(".", 1)[-1].strip("_") for n in nodes]
    stem = re.sub(r"[^0-9A-Za-z_]", "_", "_".join(shorts))[:48]
    return f"scaffold_{stem}_{stretch.start_seq}_{stretch.end_seq}"


# -- code emission -----------------------------------------------------------


@dataclass(frozen=True)
class _Ctx:
    mode: str            # multi | single
    msl: dict            # dtype name -> MSL type text (T substituted)
    n_inst: int
    rb: int              # rows per threadgroup in multi mode
    rows: str            # row count, Metal int expression


def _flit(v) -> str:
    # a double literal cast to float reproduces mlx's python-float conversion
    return f"((float){float(v)!r})"


def _offset(view: View, idxs: list[str]) -> str:
    """Flat offset into the view's base buffer; idxs right-align to the view."""
    terms = []
    for d in range(len(view.shape)):
        if view.shape[d].is_one:
            continue
        s = view.strides[d]
        ie = idxs[len(idxs) - len(view.shape) + d]
        if s.is_literal:
            if s.values[0] == 0:
                continue
            terms.append(ie if s.values[0] == 1 else f"{ie} * {s.values[0]}")
        else:
            terms.append(f"{ie} * ({s.body})")
    return " + ".join(terms) if terms else "0"


def _decomp(dims: tuple[Dim, ...], src: str) -> tuple[list[str], list[str]]:
    """Row-major index decomposition of a flat index expression."""
    if len(dims) == 0:
        return [], []
    if len(dims) == 1:
        return [], [src]
    lines = [f"int t_ = {src};"]
    idxs = [""] * len(dims)
    for d in range(len(dims) - 1, 0, -1):
        idxs[d] = f"i{d}_"
        lines.append(f"const int i{d}_ = t_ % ({dims[d].body}); t_ /= ({dims[d].body});")
    idxs[0] = "t_"
    return lines, idxs


def _flat(v: _Value, shape: tuple[Dim, ...], n_inst: int) -> bool:
    """The view reads at the same flat index the iteration writes."""
    return is_contiguous(v.view, n_inst) and shapes_equal(v.view.shape, shape)


def _indent(lines: list[str]) -> list[str]:
    return ["    " + l for l in lines]


def _emit_body(stages: list[_Stage], ctx: _Ctx) -> str:
    lines = ["const int lid_ = (int)thread_position_in_threadgroup.x;"]
    if ctx.mode == "multi":
        lines.append("const int tg_ = (int)thread_position_in_grid.y;")
        if any(st.kind in ("rms", "reduce") for st in stages):
            lines.append(f"threadgroup float sh_[{TGX}];")
    emitters = {"ew": _emit_ew, "copy": _emit_ew, "matmul": _emit_matmul,
                "rms": _emit_rms, "reduce": _emit_reduce, "qmm": _emit_qmm,
                "rope": _emit_rope}
    for i, st in enumerate(stages):
        lines.append(f"{{ // {st.op} -> {st.out.name}")
        lines.extend(_indent(emitters[st.kind](st, ctx)))
        lines.append("}")
        if i + 1 < len(stages):
            lines.append("threadgroup_barrier(mem_flags::mem_device);")
    return "\n".join(lines) + "\n"


def _row_block(inner: list[str], ctx: _Ctx) -> list[str]:
    """Wrap a per-row body so this threadgroup covers its block of rows.
    row_ is uniform across the threadgroup, so barriers inside stay legal."""
    if ctx.rb == 1:
        return ["const int row_ = tg_;", *inner]
    return [
        f"for (int rb_ = 0; rb_ < {ctx.rb}; ++rb_) {{",
        f"    const int row_ = tg_ * {ctx.rb} + rb_;",
        f"    if (row_ < ({ctx.rows})) {{",
        *_indent(_indent(inner)),
        "    }",
        "}",
    ]


def _emit_ew(st: _Stage, ctx: _Ctx) -> list[str]:
    dims = st.out.shape
    inner: list[str] = []
    values = [s for s in st.srcs if isinstance(s, _Value)]
    idxs: list[str] = []
    if any(not _flat(v, dims, ctx.n_inst) for v in values):
        dlines, idxs = _decomp(dims, "F_")
        inner.extend(dlines)
    exprs = []
    for s in st.srcs:
        if isinstance(s, _Value):
            off = "F_" if _flat(s, dims, ctx.n_inst) else _offset(s.view, idxs)
            load = f"{s.buf.name}[{off}]"
            exprs.append(load if st.kind == "copy" else f"((float){load})")
        else:
            exprs.append(_flit(s))
    fmt = "{a}" if st.kind == "copy" else st.fmt
    result = fmt.format(a=exprs[0], b=exprs[1] if len(exprs) > 1 else "")
    cast = "bool" if st.bool_out else ctx.msl[st.out.dtype]
    inner.append(f"{st.out.name}[F_] = ({cast}){result};")
    if ctx.mode == "multi":
        last = f"({dims[-1].body})" if dims else "1"
        return _row_block([
            f"for (int e_ = lid_; e_ < {last}; e_ += {TGX}) {{",
            f"    const int F_ = row_ * {last} + e_;",
            *_indent(inner),
            "}",
        ], ctx)
    numel = prod_dims(dims, ctx.n_inst)
    return [f"for (int F_ = lid_; F_ < ({numel.body}); F_ += {TGX}) {{",
            *_indent(inner), "}"]


def _emit_matmul(st: _Stage, ctx: _Ctx) -> list[str]:
    a, b = st.srcs
    dims = st.out.shape
    lead, n_dim = dims[:-1], dims[-1]
    k_dim = a.view.shape[-1]
    cast = ctx.msl[st.out.dtype]

    def dot(lead_idxs: list[str], n_idx: str, store: str) -> list[str]:
        off_a = _offset(a.view, lead_idxs + ["k_"])
        off_b = _offset(b.view, lead_idxs[:-1] + ["k_", n_idx])
        return [
            "float acc_ = 0.0f;",
            f"for (int k_ = 0; k_ < ({k_dim.body}); ++k_) {{",
            f"    acc_ += ((float){a.buf.name}[{off_a}]) * ((float){b.buf.name}[{off_b}]);",
            "}",
            f"{store} = ({cast})acc_;",
        ]

    if ctx.mode == "multi":
        if ctx.rb > 1 and len(dims) == 2:
            return _emit_matmul_blocked(st, ctx)
        dlines, lead_idxs = _decomp(lead, "row_")
        inner = dlines + [
            f"for (int n_ = lid_; n_ < ({n_dim.body}); n_ += {TGX}) {{",
            *_indent(dot(lead_idxs, "n_", f"{st.out.name}[row_ * ({n_dim.body}) + n_]")),
            "}",
        ]
        return _row_block(inner, ctx)
    numel = prod_dims(dims, ctx.n_inst)
    dlines, idxs = _decomp(dims, "F_")
    inner = dlines + dot(idxs[:-1], idxs[-1], f"{st.out.name}[F_]")
    return [f"for (int F_ = lid_; F_ < ({numel.body}); F_ += {TGX}) {{",
            *_indent(inner), "}"]


def _emit_matmul_blocked(st: _Stage, ctx: _Ctx) -> list[str]:
    """Rank-2 matmul with register blocking: each thread accumulates one
    output column for every row in the block, so B streams once per block of
    RB rows instead of once per row. Per-element accumulation order is the
    same serial k loop as the plain path. Rows past the edge clamp to the
    last row and their result is discarded at the store."""
    a, b = st.srcs
    n_dim = st.out.shape[-1]
    k_dim = a.view.shape[-1]
    cast = ctx.msl[st.out.dtype]
    rb = ctx.rb
    off_b = _offset(b.view, ["k_", "n_"])
    head = [
        f"const int r0_ = tg_ * {rb};",
        f"const int rmax_ = ({ctx.rows}) - 1;",
    ]
    for r in range(rb):
        head.append(f"const int row{r}_ = metal::min(r0_ + {r}, rmax_);")
    inner = [f"float acc{r}_ = 0.0f;" for r in range(rb)]
    inner.append(f"for (int k_ = 0; k_ < ({k_dim.body}); ++k_) {{")
    inner.append(f"    const float b_ = (float){b.buf.name}[{off_b}];")
    for r in range(rb):
        off_a = _offset(a.view, [f"row{r}_", "k_"])
        inner.append(f"    acc{r}_ += ((float){a.buf.name}[{off_a}]) * b_;")
    inner.append("}")
    for r in range(rb):
        inner.append(f"if (r0_ + {r} < ({ctx.rows})) "
                     f"{st.out.name}[(r0_ + {r}) * ({n_dim.body}) + n_] = ({cast})acc{r}_;")
    return head + [
        f"for (int n_ = lid_; n_ < ({n_dim.body}); n_ += {TGX}) {{",
        *_indent(inner),
        "}",
    ]


def _emit_qmm(st: _Stage, ctx: _Ctx) -> list[str]:
    """Affine dequant matmul, transposed weights. Accumulation is factored
    per group, scale * dot(x, q) + bias * sum(x), in fp32, matching the
    library's grouping so low bits stay close."""
    x, wq, scales, biases = st.srcs
    bits, gs = st.qmm_bits, st.qmm_group
    per = 32 // bits
    mask = (1 << bits) - 1
    dims = st.out.shape
    lead, n_dim = dims[:-1], dims[-1]
    kp = wq.view.shape[1]           # packed words per output row, literal
    groups = scales.view.shape[1]   # groups per output row, literal
    cast = ctx.msl[st.out.dtype]

    words = gs // per  # packed words per group; stage builder guarantees exact

    def dot(lead_idxs: list[str], n_idx: str, store: str) -> list[str]:
        lines = [
            "float acc_ = 0.0f;",
            f"for (int g_ = 0; g_ < ({groups.body}); ++g_) {{",
            f"    const float sc_ = (float){scales.buf.name}[{n_idx} * ({groups.body}) + g_];",
            f"    const float bi_ = (float){biases.buf.name}[{n_idx} * ({groups.body}) + g_];",
            "    float dq_ = 0.0f, dx_ = 0.0f;",
            f"    for (int w_i = 0; w_i < {words}; ++w_i) {{",
            f"        const uint p_ = (uint){wq.buf.name}[{n_idx} * ({kp.body}) + g_ * {words} + w_i];",
            f"        const int kb_ = g_ * {gs} + w_i * {per};",
        ]
        # one word load covers `per` weights; unroll its nibbles with literal shifts
        for j in range(per):
            off_x = _offset(x.view, lead_idxs + [f"(kb_ + {j})"])
            lines += [
                f"        {{ const float xv_ = (float){x.buf.name}[{off_x}];",
                f"          dq_ += xv_ * (float)((p_ >> {j * bits}) & {mask}u); dx_ += xv_; }}",
            ]
        lines += [
            "    }",
            "    acc_ += sc_ * dq_ + bi_ * dx_;",
            "}",
            f"{store} = ({cast})acc_;",
        ]
        return lines

    if ctx.mode == "multi":
        dlines, lead_idxs = _decomp(lead, "row_")
        inner = dlines + [
            f"for (int n_ = lid_; n_ < ({n_dim.body}); n_ += {TGX}) {{",
            *_indent(dot(lead_idxs, "n_", f"{st.out.name}[row_ * ({n_dim.body}) + n_]")),
            "}",
        ]
        return _row_block(inner, ctx)
    numel = prod_dims(dims, ctx.n_inst)
    dlines, idxs = _decomp(dims, "F_")
    inner = dlines + dot(idxs[:-1], idxs[-1], f"{st.out.name}[F_]")
    return [f"for (int F_ = lid_; F_ < ({numel.body}); F_ += {TGX}) {{",
            *_indent(inner), "}"]


def _emit_rope(st: _Stage, ctx: _Ctx) -> list[str]:
    """Position-dependent rotation of the leading `dims` of the last axis;
    the tail copies through. Angles use metal::precise so results are
    deterministic; the ladder compares them against the library."""
    x = st.srcs[0]
    dims_, traditional, base, scale, offset = st.rope_args
    half = dims_ // 2
    log_base = _flit(math.log(base))
    shape = st.out.shape
    lead, d_dim = shape[:-1], shape[-1]
    cast = ctx.msl[st.out.dtype]

    def body(lead_idxs: list[str], n_idx: str, d_idx: str, store: str) -> list[str]:
        off_v = _offset(x.view, lead_idxs + [d_idx])
        off_p = _offset(x.view, lead_idxs + ["dp_"])
        if traditional:
            pick = [
                f"const bool first_ = (({d_idx}) & 1) == 0;",
                f"const int j_ = ({d_idx}) >> 1;",
                f"const int dp_ = first_ ? ({d_idx}) + 1 : ({d_idx}) - 1;",
            ]
        else:
            pick = [
                f"const bool first_ = ({d_idx}) < {half};",
                f"const int j_ = first_ ? ({d_idx}) : ({d_idx}) - {half};",
                f"const int dp_ = first_ ? ({d_idx}) + {half} : ({d_idx}) - {half};",
            ]
        return [
            f"float r_;",
            f"if (({d_idx}) >= {dims_}) {{",
            f"    r_ = (float){x.buf.name}[{off_v}];",
            "} else {",
            *_indent(pick),
            f"    const float pos_ = ((float)({n_idx}) + {_flit(offset)}) * {_flit(scale)};",
            f"    const float th_ = pos_ * metal::precise::exp(-((float)(2 * j_) / {_flit(dims_)}) * {log_base});",
            "    const float c_ = metal::precise::cos(th_);",
            "    const float s_ = metal::precise::sin(th_);",
            f"    const float xv_ = (float){x.buf.name}[{off_v}];",
            f"    const float xp_ = (float){x.buf.name}[{off_p}];",
            "    r_ = first_ ? (xv_ * c_ - xp_ * s_) : (xp_ * s_ + xv_ * c_);",
            "}",
            f"{store} = ({cast})r_;",
        ]

    if ctx.mode == "multi":
        dlines, lead_idxs = _decomp(lead, "row_")
        inner = dlines + [
            f"for (int d_ = lid_; d_ < ({d_dim.body}); d_ += {TGX}) {{",
            *_indent(body(lead_idxs, lead_idxs[-1],
                          "d_", f"{st.out.name}[row_ * ({d_dim.body}) + d_]")),
            "}",
        ]
        return _row_block(inner, ctx)
    numel = prod_dims(shape, ctx.n_inst)
    dlines, idxs = _decomp(shape, "F_")
    inner = dlines + body(idxs[:-1], idxs[-2], idxs[-1], f"{st.out.name}[F_]")
    return [f"for (int F_ = lid_; F_ < ({numel.body}); F_ += {TGX}) {{",
            *_indent(inner), "}"]


def _row_reduce_tree(comb_fmt: str) -> list[str]:
    comb = comb_fmt.format(a="sh_[lid_]", x="sh_[lid_ + s_]")
    return [
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        f"for (int s_ = {TGX // 2}; s_ > 0; s_ >>= 1) {{",
        f"    if (lid_ < s_) sh_[lid_] = {comb};",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "}",
    ]


def _row_offset(x: _Value, n_inst: int) -> tuple[list[str], str]:
    """Offset of element (row_, j_) of a row-shaped operand, with any index
    decomposition lines it needs. row_ is the flat index over the lead dims."""
    if is_contiguous(x.view, n_inst):
        return [], f"row_ * ({x.view.shape[-1].body}) + j_"
    dlines, lead_idxs = _decomp(x.view.shape[:-1], "row_")
    return dlines, _offset(x.view, lead_idxs + ["j_"])


def _emit_rms(st: _Stage, ctx: _Ctx) -> list[str]:
    x, w = st.srcs
    d = x.view.shape[-1].body
    cast = ctx.msl[st.out.dtype]
    eps = _flit(st.eps)
    dlines, off = _row_offset(x, ctx.n_inst)
    off_w = _offset(w.view, ["j_"])
    store = f"{st.out.name}[row_ * ({d}) + j_]"
    normed = f"((float){x.buf.name}[{off}]) * scale_ * ((float){w.buf.name}[{off_w}])"
    if ctx.mode == "multi":
        return _row_block(dlines + [
            "float p_ = 0.0f;",
            f"for (int j_ = lid_; j_ < ({d}); j_ += {TGX}) {{",
            f"    const float v_ = (float){x.buf.name}[{off}];",
            "    p_ += v_ * v_;",
            "}",
            "sh_[lid_] = p_;",
            *_row_reduce_tree("({a} + {x})"),
            f"const float scale_ = metal::precise::rsqrt(sh_[0] / ((float)({d})) + {eps});",
            f"for (int j_ = lid_; j_ < ({d}); j_ += {TGX}) {{",
            f"    {store} = ({cast})({normed});",
            "}",
        ], ctx)
    rows = prod_dims(x.view.shape[:-1], ctx.n_inst)
    inner = dlines + [
        "float p_ = 0.0f;",
        f"for (int j_ = 0; j_ < ({d}); ++j_) {{",
        f"    const float v_ = (float){x.buf.name}[{off}];",
        "    p_ += v_ * v_;",
        "}",
        f"const float scale_ = metal::precise::rsqrt(p_ / ((float)({d})) + {eps});",
        f"for (int j_ = 0; j_ < ({d}); ++j_) {{",
        f"    {store} = ({cast})({normed});",
        "}",
    ]
    return [f"for (int row_ = lid_; row_ < ({rows.body}); row_ += {TGX}) {{",
            *_indent(inner), "}"]


def _emit_reduce(st: _Stage, ctx: _Ctx) -> list[str]:
    x = st.srcs[0]
    init, comb_fmt, divide = _REDUCE[st.reduce_op]
    d = x.view.shape[-1].body
    cast = ctx.msl[st.out.dtype]
    dlines, off = _row_offset(x, ctx.n_inst)
    comb = comb_fmt.format(a="a_", x=f"((float){x.buf.name}[{off}])")

    def final(acc: str) -> str:
        return f"({acc} / ((float)({d})))" if divide else acc

    if ctx.mode == "multi":
        return _row_block(dlines + [
            f"float a_ = {init};",
            f"for (int j_ = lid_; j_ < ({d}); j_ += {TGX}) a_ = {comb};",
            "sh_[lid_] = a_;",
            *_row_reduce_tree(comb_fmt),
            f"if (lid_ == 0) {st.out.name}[row_] = ({cast}){final('sh_[0]')};",
        ], ctx)
    rows = prod_dims(x.view.shape[:-1], ctx.n_inst)
    inner = dlines + [
        f"float a_ = {init};",
        f"for (int j_ = 0; j_ < ({d}); ++j_) a_ = {comb};",
        f"{st.out.name}[row_] = ({cast}){final('a_')};",
    ]
    return [f"for (int row_ = lid_; row_ < ({rows.body}); row_ += {TGX}) {{",
            *_indent(inner), "}"]


def lower_naive(
    trace: Trace,
    stretch: Stretch,
    instances: Sequence[Sequence[Sequence[int]]] = (),
) -> KernelSpec | None:
    """Lower one region stretch to a single correct Metal kernel.

    instances are the stretch's boundary input shapes at other traced sizes
    (stretch_input_shapes over locate_span results), input_ids order; the
    primary trace is always instance zero. Dims that vary across instances
    become inK.shape[j] launch-grammar accessors; constant dims may bake as
    literals. Raises NoScaffold with a named reason when the op sequence has
    no naive lowering; the caller then skips the region. Output order: the
    region outputs (stretch.output_ids order, named out0..) come first, then
    tmpN scratch buffers the caller ignores.
    """
    if not trace.nodes or stretch.end_seq < stretch.start_seq:
        raise NoScaffold("empty-region")
    return _Lowering(trace, stretch, list(instances)).run()
