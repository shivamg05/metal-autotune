"""Gate 1: static checks on a KernelSpec against the region's IO contract.
No GPU work runs here.

Tolerance secrecy is structural: the contract carries names, ranks, dtypes,
and liveness only, so this module cannot see or leak tolerance values.

Input ranks have no declaration on the kernel side; they are enforced through
the launch grammar instead, by probe-evaluating every expression against
dummy shapes of the contract's ranks so a bad input index or axis fails here,
not at dispatch.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from autotuner_runtime.grammar import Expr, GrammarError
from autotuner_runtime.kernels import _DTYPES, KernelSpec

# The kernel name is pasted into the generated signature; a bad name fails
# only at probe eval, so it is gated here.
_C_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TEMPLATE_INPUT_REF = re.compile(r"^in\d+$")
METAL_BUFFER_LIMIT = 31  # Metal argument-table buffer indices are 0..30
# Deliberately conservative policy for the shared display GPU. A subprocess
# cannot isolate a GPU hang from WindowServer. This catches obvious oversized
# serial launches; it does not establish that arbitrary Metal source is safe.
SINGLE_GROUP_ELEMENT_LIMIT = 1 << 20
THREADGROUP_THREAD_LIMIT = 1024


def buffer_count(spec: KernelSpec, input_ranks) -> int:
    """Match MLX 0.32.2's generated data and input-metadata buffer arguments."""
    if spec.native_call is not None:
        native = spec.native_call
        input_ranks = [input_ranks[b["array"]] if "array" in b else 0 for b in native["bindings"]]
        names = native["factory"]["input_names"]
    else:
        names = spec.input_names
    return len(names) + len(spec.output_names) + sum(
        name + suffix in spec.source
        for name, rank in zip(names, input_ranks) if rank > 0
        for suffix in ("_shape", "_strides", "_ndim"))


@dataclass(frozen=True)
class RegionContract:
    """The region's boundary as gate 1 sees it: input/output names, ranks,
    dtypes, and which outputs are live. No tensors, no tolerances."""

    input_names: tuple[str, ...]
    input_ranks: tuple[int, ...]
    input_dtypes: tuple[str, ...]
    output_names: tuple[str, ...]
    output_ranks: tuple[int, ...]
    output_dtypes: tuple[str, ...]
    live_outputs: tuple[str, ...]
    requires_fallback: bool = False   # the hypothesis is shape-specialized
    input_shapes: tuple[tuple[int, ...], ...] = ()   # at the traced size; expressions are evaluated on them
    output_shapes: tuple[tuple[int, ...], ...] = ()
    native_call: dict | None = None
    input_signature: list | None = None
    input_signatures: list | None = None


@dataclass(frozen=True)
class Failure:
    check: str
    detail: str


def launch_resource_failure(spec: KernelSpec, input_shapes) -> Failure | None:
    """Check one concrete dispatch without compiling or interpreting Metal.

    All inputs and all allocated outputs (including scratch) count. Tiled
    launches are exempt from the serial-buffer policy. Call for every eval
    shape, since a launch can be small at one size and oversized at another.
    """
    if spec.reference_sequence is not None:
        return None  # this trusted starter replays original dispatches; it is not one GPU launch
    if spec.stages:
        try:
            if spec.fallback_predicate is not None and Expr(spec.fallback_predicate).evaluate(input_shapes):
                return None
            from autotuner_runtime.stages import shapes
            stages, outputs = shapes(spec, input_shapes, [None] * len(input_shapes))
            for i, (kernel, inputs, _) in enumerate(stages):
                failure = launch_resource_failure(kernel, [s for s, _ in inputs])
                if failure is not None:
                    return Failure(failure.check, f"stage {i}: {failure.detail}")
            expected = [tuple(Expr(e).evaluate(input_shapes) for e in shape) for shape in spec.output_shapes]
            if outputs != list(zip(expected, spec.output_dtypes)):
                return Failure("stage_boundary", "stage results must match every region output shape and dtype")
        except (ValueError, ArithmeticError) as exc:
            return Failure("stages", str(exc))
        return None
    try:
        if spec.fallback_predicate is not None and Expr(spec.fallback_predicate).evaluate(input_shapes):
            return None  # the runtime calls the library for this shape
        axes = {}
        for label, exprs in (("grid", spec.grid), ("threadgroup", spec.threadgroup)):
            if len(exprs) != 3:
                return Failure("launch_arity", f"{label} needs three dimensions")
            values = tuple(Expr(e).evaluate(input_shapes) for e in exprs)
            if any(type(v) is not int or v <= 0 for v in values):
                return Failure("launch_extent", f"{label} must contain positive integers, got {values}")
            axes[label] = values
        threads = math.prod(axes["threadgroup"])
        if threads > THREADGROUP_THREAD_LIMIT:
            return Failure("launch_extent", f"threadgroup has {threads} threads; limit is {THREADGROUP_THREAD_LIMIT}")
        outputs = [tuple(Expr(e).evaluate(input_shapes) for e in shape)
                   for shape in spec.output_shapes]
        if any(type(v) is not int or v < 0 for shape in outputs for v in shape):
            return Failure("output_extent", "output dimensions must be nonnegative integers")
    except (GrammarError, ArithmeticError) as exc:
        return Failure("launch_grammar", str(exc))
    groups = math.prod((g + t - 1) // t for g, t in zip(axes["grid"], axes["threadgroup"]))
    if groups == 1:
        buffers = list(zip(spec.input_names, input_shapes)) + list(zip(spec.output_names, outputs))
        for name, shape in buffers:
            elements = math.prod(shape)
            if elements > SINGLE_GROUP_ELEMENT_LIMIT:
                return Failure(
                    "serial_launch", f"one threadgroup with buffer {name} containing {elements:,} elements "
                    f"exceeds the {SINGLE_GROUP_ELEMENT_LIMIT:,}-element serial-launch policy; "
                    "tile the work across threadgroups or choose a smaller region")
    return None


def check(spec: KernelSpec, contract: RegionContract) -> list[Failure]:
    """Every static failure in the spec, named. Empty list means gate 1 passes."""
    fails: list[Failure] = []
    if spec.input_signature != contract.input_signature:
        return [Failure("input_signature", "input specialization differs from the region contract")]
    if spec.input_signatures != contract.input_signatures:
        return [Failure("input_signatures", "input specializations differ from the region contract")]
    if spec.native_call != contract.native_call:
        return [Failure("native_contract", "native call settings differ from the original kernel")]
    if spec.native_call is not None and not spec.stages and (spec.template or len(spec.output_names) != len(contract.output_names)):
        return [Failure("native_contract", "native templates and output allocation count are fixed")]
    if spec.native_call is not None and not spec.stages:
        factory = spec.native_call["factory"]
        if (spec.ensure_row_contiguous != factory.get("ensure_row_contiguous", True)
                or spec.atomic_outputs != factory.get("atomic_outputs", False)):
            return [Failure("native_contract", "native layout and atomic settings are fixed")]
    buffers = buffer_count(spec, contract.input_ranks)
    if spec.reference_sequence is None and not spec.stages and buffers > METAL_BUFFER_LIMIT:
        fails.append(Failure("buffer_limit", f"kernel needs {buffers} Metal buffer arguments; "
                             f"the limit is {METAL_BUFFER_LIMIT}, including input metadata and scratch"))

    if not _C_IDENTIFIER.match(spec.name):
        fails.append(Failure(
            "kernel_name",
            f"{spec.name!r} is not a valid C identifier; it is pasted into the "
            "generated signature and would fail only at probe eval",
        ))

    if tuple(spec.input_names) != tuple(contract.input_names):
        fails.append(Failure(
            "input_names",
            f"kernel inputs {tuple(spec.input_names)} != region inputs "
            f"{tuple(contract.input_names)}",
        ))
    # region outputs come first; extra scratch buffers are legal only when
    # every one is named tmp<N> (naive scaffolds stage through device memory)
    n_region = len(contract.output_names)
    head = tuple(spec.output_names[:n_region])
    extras = tuple(spec.output_names[n_region:])
    if head != tuple(contract.output_names) or not all(
        e.startswith("tmp") and e[3:].isdigit() for e in extras
    ):
        fails.append(Failure(
            "output_names",
            f"kernel outputs {tuple(spec.output_names)} must start with region "
            f"outputs {tuple(contract.output_names)}, extras all tmp<N>",
        ))
    for name in contract.live_outputs:
        if name not in spec.output_names:
            fails.append(Failure(
                "live_output_dropped",
                f"live value {name!r} is neither produced nor listed as an output",
            ))

    n_out = len(spec.output_names)
    if len(spec.output_shapes) != n_out or len(spec.output_dtypes) != n_out:
        fails.append(Failure(
            "output_arity",
            f"{n_out} output names but {len(spec.output_shapes)} shape tuples "
            f"and {len(spec.output_dtypes)} dtypes",
        ))
    for name, dt, want in zip(spec.output_names, spec.output_dtypes, contract.output_dtypes):
        if dt != want:
            fails.append(Failure(
                "output_dtype",
                f"output {name!r}: {dt} != frozen boundary dtype {want}",
            ))
    # zips stop at the region outputs; tmp extras carry no boundary contract
    for name, exprs, want in zip(spec.output_names, spec.output_shapes, contract.output_ranks):
        if len(exprs) != want:
            fails.append(Failure(
                "output_rank",
                f"output {name!r}: {len(exprs)} shape expressions, region rank is {want}",
            ))

    shapes = [tuple(s) for s in contract.input_shapes] or [(4,) * r for r in contract.input_ranks]
    for axis_name, exprs in (("grid", spec.grid), ("threadgroup", spec.threadgroup)):
        if len(exprs) != 3:
            fails.append(Failure(
                "launch_arity", f"{axis_name} has {len(exprs)} expressions, needs 3"))
        for i, text in enumerate(exprs):
            _probe(f"{axis_name}[{i}]", text, shapes, fails)
    for j, exprs in enumerate(spec.output_shapes):
        for i, text in enumerate(exprs):
            _probe(f"output_shapes[{j}][{i}]", text, shapes, fails)
    # a region output's expressions must give the shape the trace recorded,
    # or the child would crash on the mismatch instead of naming a gate
    for name, exprs, want in zip(spec.output_names, spec.output_shapes, contract.output_shapes):
        try:
            got = tuple(Expr(e).evaluate(shapes) for e in exprs)
        except (GrammarError, ArithmeticError):
            continue  # already reported by the probe above
        if got != tuple(want):
            fails.append(Failure(
                "output_shape",
                f"output {name!r}: the expressions give {got} at the traced size, "
                f"the region records {tuple(want)}",
            ))
    if spec.fallback_predicate is not None:
        _probe("fallback_predicate", spec.fallback_predicate, shapes, fails)
        if contract.input_shapes:
            try:
                if Expr(spec.fallback_predicate).evaluate(contract.input_shapes):
                    fails.append(Failure(
                        "fallback_on_primary",
                        "fallback_predicate is true on the primary workload: true means "
                        "run the original library, not the custom kernel. Make it false "
                        "on the shapes being optimized and true only on unsupported shapes.",
                    ))
            except (GrammarError, ArithmeticError):
                pass  # expression diagnostics are reported by the checks above
    elif contract.requires_fallback:
        fails.append(Failure(
            "fallback_missing",
            "shape-specialized kernel declares no fallback predicate",
        ))

    for tname, dt in spec.template:
        if _TEMPLATE_INPUT_REF.match(dt):
            if int(dt[2:]) >= len(spec.input_names):
                fails.append(Failure(
                    "template",
                    f"template {tname!r} references {dt} but the kernel has "
                    f"{len(spec.input_names)} inputs",
                ))
        elif dt not in _DTYPES:
            fails.append(Failure(
                "template", f"template {tname!r}: unknown dtype {dt!r}"))

    # Dummy rank probes do not describe the actual amount of work. Apply
    # the resource policy only to concrete shapes supplied by the caller.
    if contract.input_shapes:
        failure = launch_resource_failure(spec, contract.input_shapes)
        if failure is not None and failure not in fails:
            fails.append(failure)
    if spec.stages:
        try:
            from autotuner_runtime.stages import shapes as stage_shapes
            stages, _ = stage_shapes(spec, shapes, contract.input_dtypes)
            for i, (kernel, inputs, outputs) in enumerate(stages):
                child = RegionContract(
                    input_names=kernel.input_names, input_ranks=tuple(len(s) for s, _ in inputs),
                    input_dtypes=tuple(dt for _, dt in inputs), output_names=kernel.output_names,
                    output_ranks=tuple(len(s) for s in outputs), output_dtypes=kernel.output_dtypes,
                    live_outputs=kernel.output_names, input_shapes=tuple(s for s, _ in inputs),
                    output_shapes=tuple(outputs))
                fails.extend(Failure(f.check, f"stage {i}: {f.detail}") for f in check(kernel, child))
        except (ValueError, ArithmeticError) as exc:
            fails.append(Failure("stages", str(exc)))
    return fails


def _probe(slot: str, text: str, dummy_shapes: list[tuple[int, ...]],
           fails: list[Failure]) -> None:
    """Parse, then evaluate against dummy shapes of the contract's ranks so a
    bad input index or axis fails statically."""
    try:
        expr = Expr(text)
    except GrammarError as e:
        fails.append(Failure("launch_grammar", f"{slot}: {e}"))
        return
    try:
        expr.evaluate(dummy_shapes)
    except GrammarError as e:
        fails.append(Failure("launch_grammar", f"{slot}: {e}"))
    except ArithmeticError:
        pass  # value-dependent (dummy sizes hit a zero divisor); not static
