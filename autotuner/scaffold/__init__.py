"""Scaffolds: the correct starting kernels the harness builds.

build_scaffold is the one entry point: stitched from the wheel's own Metal
source when the region is a single library op we can wrap verbatim (library
parity by construction), naive lowering otherwise."""

from .lower import (
    _CONSTANT_NAMES, _POWER_NAMES, _SOFTMAX_NAMES, _LOGSUMEXP_NAMES,
    _CAST_NAMES, _CONCAT_NAMES, _EW_NAMES, _GETITEM_NAMES, _LN_NAMES, _MATMUL_NAMES,
    _QMM_NAMES, _REDUCE_NAMES, _RMS_NAMES, _ROPE_NAMES, _SPLIT_NAMES, _STACK_NAMES,
    _VIEW_NAMES, lower_naive, stretch_input_shapes,
)
from .stitch import stitch_affine_qmm_t, stitch_qmm_chain, stitch_quantized_matmul
from .symshape import NoScaffold

_COVERED = (frozenset(_EW_NAMES) | frozenset(_REDUCE_NAMES) | _MATMUL_NAMES | _RMS_NAMES
            | _LN_NAMES | _QMM_NAMES | _ROPE_NAMES | _VIEW_NAMES | _CAST_NAMES
            | _GETITEM_NAMES | _SPLIT_NAMES | _CONCAT_NAMES | _STACK_NAMES
            | _CONSTANT_NAMES | _POWER_NAMES | _SOFTMAX_NAMES | _LOGSUMEXP_NAMES)


def uncovered_op(ops) -> str | None:
    """The first op no scaffold path can write, or None. Cheap enough to run
    before a region is captured and priced."""
    return next((op for op in ops if op not in _COVERED), None)


def build_scaffold(trace, stretch, instances=()):
    nodes = trace.nodes[stretch.start_seq:stretch.end_seq + 1]
    if len(nodes) == 1 and nodes[0].kernel_definition is not None:
        from .native import native_seed
        return native_seed(trace, stretch)
    if any(node.kernel_definition is not None for node in nodes):
        from .native import reference_sequence_seed
        return reference_sequence_seed(trace, stretch)
    if len(nodes) == 1 and nodes[0].op == "mx.quantized_matmul" \
            and tuple(stretch.input_ids) == tuple(nodes[0].in_arrays) \
            and len(stretch.output_ids) == 1 and len(nodes[0].in_specs) == 4:
        kw = nodes[0].scalar_args.get("kwargs", {})
        if kw.get("transpose", True) is True and kw.get("mode", "affine") == "affine":
            gs, b = kw.get("group_size", 64), kw.get("bits", 4)
            # qmv for decode, qmm_t for prefill; each refuses the other's range
            for stitch in (stitch_quantized_matmul, stitch_affine_qmm_t):
                try:
                    return stitch(*nodes[0].in_specs[:4], group_size=gs, bits=b)
                except NoScaffold:
                    pass  # try the next kernel, then the naive path
    if len(nodes) > 1 and nodes[0].op == "mx.quantized_matmul":
        try:
            return stitch_qmm_chain(nodes, stretch.input_ids, stretch.output_ids)
        except NoScaffold:
            pass  # the naive path still gets its chance
    return lower_naive(trace, stretch, instances)


__all__ = ["build_scaffold", "lower_naive", "stitch_affine_qmm_t", "stitch_qmm_chain",
           "stitch_quantized_matmul", "stretch_input_shapes", "uncovered_op", "NoScaffold"]
