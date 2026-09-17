"""Legacy fp32 reference utilities; new searches use original-output tolerances.

Two references, one rule. At the region (gate 8) the golden is the region's
own recorded ops replayed with float bindings promoted to fp32 and quantized
ops routed through a substitution table (dequantize + matmul in fp32). For
the whole model it is the untouched model itself run at fp32
(fp32_reference), with its mutable state isolated. Unverifiable compiled
sections are refused instead of being treated as a high-precision oracle.
Candidates and the library are both scored against the golden:
err(candidate) <= kappa * err(library) + floor, kappa and floor caller-held.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn

from autotuner.trace.replay import replay
from autotuner.trace.types import TraceNode

from .numeric import FLOAT_DTYPES, nonfinite_pattern_ok


def fp32_reference(model, inputs, tracer, *, check_graph=True) -> list[mx.array]:
    """The model's own math run at fp32, as the reference a reordering edit is
    graded against. Every float array the call reaches is lifted to fp32 for
    the duration and restored after: the parameters, any module buffer, the
    inputs, and every state object (a KV cache) it reads or writes. Packed
    quantized weights are ints, so they stay packed and their matmuls unpack
    on the fly in fp32; that is what keeps this small on an 8B model. The run
    is recorded, and refused if any op handed back a lower precision, so a
    forward that casts to a fixed low precision inside itself cannot pass off
    a rounded answer as the reference. Live state is never touched: a cache
    write lands on the lifted copy and the original comes back untouched.
    For decode this is the same prefix state promoted to fp32, not a new
    prefix generated in fp32: both candidates start from the same KV inputs.
    """
    from mlx.utils import tree_map
    from ..trace.walk import flatten_arrays, state_holders, reachable

    def lift(value):
        if isinstance(value, mx.array):
            promoted = value.astype(mx.float32) if value.dtype in FLOAT_DTYPES else value
            # astype(fp32) may return the same object. MLX indexed assignment
            # changes that object's graph, so mutable buffers need a new array
            # object even when their dtype already matches (including ints).
            return mx.array(promoted)
        if isinstance(value, nn.Module):
            return value  # children are promoted through model.parameters()
        if isinstance(value, list):
            return [lift(v) for v in value]
        if isinstance(value, tuple):
            return tuple(lift(v) for v in value)
        if isinstance(value, dict):
            return {k: lift(v) for k, v in value.items()}
        return value

    params = model.parameters() if isinstance(model, nn.Module) else {}
    modules = [m for _p, m in model.named_modules()] if isinstance(model, nn.Module) else []
    buffers = [(m, k, v) for m in modules for k, v in list(m.items())
               if k.startswith("_") and isinstance(v, (mx.array, list, tuple, dict))]
    holders = [obj for _p, obj in state_holders(model)]
    saved = [dict(vars(obj)) for obj in holders]
    try:
        if params:
            model.update(tree_map(lift, params))
        for m, k, v in buffers:
            m[k] = lift(v)
        for obj in holders:
            for k, v in list(vars(obj).items()):
                setattr(obj, k, lift(v))
        lifted_inputs = [lift(a) for a in inputs]
        seeds = [a for _, a in reachable(model) if isinstance(a, mx.array)] + lifted_inputs
        with _audit_fp32_ops(tracer.recorder, retain_graph=check_graph, seeds=seeds) as check_outputs:
            if check_graph:
                _trace, outs = tracer.trace(model, lifted_inputs)
            else:
                # An advancing sequence creates isolated cache copies inside
                # the call. They need no replay graph, but every recorded op
                # still goes through the same precision/opaque-call audit.
                tracer.recorder.arm(model, lifted_inputs, {})
                try:
                    outs = model(*lifted_inputs)
                finally:
                    tracer.recorder.disarm()
                check_outputs(flatten_arrays(outs))
        outputs = flatten_arrays(outs)
        mx.eval(outputs)
        return outputs
    finally:
        if not check_graph:
            tracer.recorder._reset_pass()  # release the lifted prefix and parameters too
        if params:
            model.update(params)
        for m, k, v in buffers:
            m[k] = v
        for obj, state in zip(holders, saved):
            vars(obj).clear()
            vars(obj).update(state)


# Audited MLX activation formulas preserve their floating input precision.
# Arbitrary compiled functions can cast internally and then return fp32, so
# their output dtype alone is not enough to certify a reference.
_FP32_ACTIVATIONS = frozenset("""
    sigmoid relu relu2 relu6 leaky_relu log_softmax elu softmax softplus
    softsign softshrink celu silu log_sigmoid gelu gelu_approx gelu_fast_approx
    step selu prelu mish hardswish hard_tanh hard_shrink softmin
""".split())


@contextmanager
def _audit_fp32_ops(recorder, *, retain_graph=True, seeds=()):
    from ..trace.recorder import OPAQUE_OP, UNNAMED_KERNEL_OP, compiled_path, kernel_path

    import weakref
    from ..trace.walk import flatten_arrays

    known = {id(a): weakref.ref(a) for a in seeds}

    def check_known(arrays):
        for a in arrays:
            ref = known.get(id(a))
            if ref is None or ref() is not a:
                raise ValueError("an unrecorded operation produced an array; no audited fp32 reference exists")

    original = recorder._append_node
    had_override = "_append_node" in vars(recorder)

    def checked(op_name, args, kwargs, out_objs, receiver=None):
        path = compiled_path(op_name)
        trusted_activation = (path is not None and path.rsplit(".", 1)[0] in
                              ("mlx.nn", "mlx.nn.layers.activations") and
                              path.rsplit(".", 1)[-1] in _FP32_ACTIVATIONS)
        # mlx-lm's compiled SwiGLU is nn.silu(gate) * x, with no dtype
        # conversion. Llama and Qwen use it instead of the nn helper directly.
        trusted_activation |= path == "mlx_lm.models.activations.swiglu"
        if (op_name in (OPAQUE_OP, UNNAMED_KERNEL_OP, "custom_kernel") or kernel_path(op_name)
                or (path and not trusted_activation)):
            raise ValueError(f"{op_name} hides its arithmetic, so no audited fp32 reference exists")
        if not retain_graph:
            check_known(flatten_arrays((args, kwargs)))
        for out in out_objs:
            if out.dtype in (mx.bfloat16, mx.float16):
                raise ValueError(f"{op_name} cannot run its math in fp32: "
                                 f"it handed back {out.dtype}, so no fp32 reference exists for this model")
        if not retain_graph:
            for out in out_objs:
                known[id(out)] = weakref.ref(out)
        if retain_graph:
            return original(op_name, args, kwargs, out_objs, receiver=receiver)

    # Observe operations before state_exit folds their nodes into one state
    # call. Otherwise a cache method can hide fp16 math behind an fp32 output.
    recorder._append_node = checked
    # Existing module/state wrappers may still call these hooks. Precision
    # auditing needs the inner op callbacks, not scope templates or cache
    # snapshots that keep each token's old buffers alive.
    hooks = {}
    if not retain_graph:
        for name in ("module_enter", "module_exit", "module_abort",
                     "state_enter", "state_exit", "state_abort"):
            hooks[name] = (name in vars(recorder), getattr(recorder, name))
            setattr(recorder, name, lambda *a, **kw: None)
    try:
        yield check_known
    finally:
        for name, (overridden, method) in hooks.items():
            if overridden:
                setattr(recorder, name, method)
            else:
                delattr(recorder, name)
        if had_override:
            recorder._append_node = original
        else:
            del recorder._append_node


DEFAULT_KAPPA = 1.25
DEFAULT_DENOM_CLAMP = 1e-6


def _quantized_matmul_fp32(x, w, scales, biases=None, transpose=True,
                           group_size=None, bits=None, mode="affine", *,
                           stream=None):
    """mx.quantized_matmul -> dequantize + matmul in fp32. group_size, bits,
    and mode flow through exactly as recorded; a recorded None hits the same
    library defaults quantized_matmul itself resolves (verified: the defaults
    of quantize/dequantize/quantized_matmul agree on this mlx version)."""
    scales = scales.astype(mx.float32)
    if biases is not None:
        biases = biases.astype(mx.float32)
    wf = mx.dequantize(w, scales, biases, group_size=group_size, bits=bits, mode=mode)
    xf = x.astype(mx.float32)
    return mx.matmul(xf, mx.swapaxes(wf, -1, -2) if transpose else wf)


_DEFAULT_TABLE: dict[str, Callable] = {
    "mx.quantized_matmul": _quantized_matmul_fp32,
}


def substitution_table(extra: Mapping[str, Callable] | None = None) -> dict[str, Callable]:
    """The quantized-op substitution table, extensible as traced ops require."""
    table = dict(_DEFAULT_TABLE)
    if extra:
        table.update(extra)
    return table


def promote_bindings(bindings: Mapping[int, mx.array]) -> dict[int, mx.array]:
    """Float bindings to fp32 (exact for fp16/bf16); everything else, including
    packed quantized weights, untouched."""
    return {
        aid: a.astype(mx.float32) if a.dtype in FLOAT_DTYPES and a.dtype != mx.float32 else a
        for aid, a in bindings.items()
    }


def golden_outputs(
    nodes: Sequence[TraceNode],
    bindings: Mapping[int, mx.array],
    outputs: Iterable[int],
    substitutions: Mapping[str, Callable] | None = None,
) -> dict[int, mx.array]:
    """Replay the span with promoted bindings and the substitution table."""
    table = substitution_table() if substitutions is None else dict(substitutions)
    for node in nodes:
        if node.kernel_definition is not None and node.op not in table:
            raise ValueError("captured Metal source has no audited fp32 reference")
    return replay(nodes, promote_bindings(bindings), outputs, op_substitute=table)


def err(
    candidates: Sequence[mx.array],
    goldens: Sequence[mx.array],
    denom_clamp: float = DEFAULT_DENOM_CLAMP,
) -> float:
    """Worst per-output max relative error against the golden, denominator
    clamped. Where the golden is non-finite: 0 if the candidate reproduces the
    pattern, inf if not."""
    worst = 0.0
    for c, g in zip(candidates, goldens, strict=True):
        if tuple(c.shape) != tuple(g.shape):
            raise ValueError(f"output shape {tuple(c.shape)} != golden {tuple(g.shape)}")
        if c.size == 0:
            continue
        cf, gf = c.astype(mx.float32), g.astype(mx.float32)
        ok = nonfinite_pattern_ok(cf, gf)
        rel = mx.abs(cf - gf) / mx.maximum(mx.abs(gf), denom_clamp)
        rel = mx.where(
            mx.isfinite(gf) & ok, rel,
            mx.where(ok, mx.zeros_like(rel), mx.array(float("inf"))),
        )
        worst = max(worst, float(mx.max(rel).item()))
    return worst


def passes(candidate_err: float, library_err: float,
           kappa: float = DEFAULT_KAPPA, floor: float = 0.0) -> bool:
    """The assoc-changing gate: reordering accumulation is allowed, being
    sloppier than the library is not."""
    return candidate_err <= kappa * library_err + floor
