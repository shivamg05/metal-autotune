"""Validate explicit stage dataflow without evaluating tensors or Metal code."""
import re

from .grammar import Expr


def wiring(spec):
    """Return buffer-table indices, rejecting writes to inputs and dead work."""
    if spec.reference_sequence is not None:
        raise ValueError("stages cannot also be an original reference sequence")
    available = {name: i for i, name in enumerate(spec.input_names)}
    produced, consumed, indices = set(), set(), []
    for i, stage in enumerate(spec.stages):
        kernel = stage.kernel
        if kernel.stages or kernel.reference_sequence is not None or kernel.native_call is not None:
            raise ValueError(f"stage {i}: must be a plain, non-nested Metal dispatch")
        if (kernel.fallback_predicate is not None or kernel.input_signature is not None
                or kernel.input_signatures is not None or kernel.atomic_outputs):
            raise ValueError(f"stage {i}: fallback and compiler policies belong to the whole candidate")
        if any(name not in available for name in stage.inputs):
            raise ValueError(f"stage {i}: inputs must name region inputs or earlier results")
        if kernel.input_names != tuple(f"in{j}" for j in range(len(stage.inputs))):
            raise ValueError(f"stage {i}: kernel input names must use the local inN ABI")
        if not stage.outputs or kernel.output_names != tuple(f"out{j}" for j in range(len(stage.outputs))):
            raise ValueError(f"stage {i}: kernel output names must use the local outN ABI")
        if len(kernel.output_shapes) != len(stage.outputs) or len(kernel.output_dtypes) != len(stage.outputs):
            raise ValueError(f"stage {i}: each output needs a shape and dtype")
        indices.append(tuple(available[name] for name in stage.inputs))
        consumed.update(stage.inputs)
        for name in stage.outputs:
            if name in available or (name not in spec.output_names and not re.fullmatch(r"tmp\d+", name)):
                raise ValueError(f"stage {i}: output {name!r} must be a new tmpN or region output")
            available[name] = len(available)
            produced.add(name)
    if not set(spec.output_names) <= produced:
        raise ValueError("stages must produce every region output")
    if produced - consumed - set(spec.output_names):
        raise ValueError("unused stage outputs would hide work from timing; remove them")
    return indices, tuple(available[name] for name in spec.output_names)


def shapes(spec, input_shapes, input_dtypes):
    """Walk stage shapes in dependency order; grammar inN is local to a stage."""
    indices, outputs = wiring(spec)
    buffers = list(zip(input_shapes, input_dtypes))
    stages = []
    for stage, inputs in zip(spec.stages, indices):
        ins = [buffers[i] for i in inputs]
        env = [shape for shape, _ in ins]
        outs = [tuple(Expr(e).evaluate(env) for e in shape) for shape in stage.kernel.output_shapes]
        stages.append((stage.kernel, ins, outs))
        buffers.extend(zip(outs, stage.kernel.output_dtypes))
    return stages, [buffers[i] for i in outputs]
