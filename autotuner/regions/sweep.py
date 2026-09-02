"""Sweep instances: find a region's node span in a retrace at
another size. The bridge from named dims to concrete shapes is retracing, not
symbolic shapes: addresses are stable across sizes for the same model, so
(op, module_address, position_in_module) locates every member, and the matched
span carries the correct shapes and shape-derived scalar args for free."""

from __future__ import annotations

from ..trace.types import Trace
from .build import _boundary
from .types import Stretch


class SweepDivergence(RuntimeError):
    """The model's op stream differs at this size: a branch went the other
    way. The enclosing scope fails the replayability screen; this exception is
    the screen's evidence."""


def locate_span(priced: Trace, stretch: Stretch, retrace: Trace, workload: str) -> Stretch:
    index = {
        (n.op, n.module_address, n.position_in_module): n for n in retrace.nodes
    }
    new_seqs = []
    for node in priced.nodes[stretch.start_seq:stretch.end_seq + 1]:
        key = (node.op, node.module_address, node.position_in_module)
        if key not in index:
            raise SweepDivergence(
                f"no match for {node.op!r} at {node.module_address!r} "
                f"pos {node.position_in_module} in the retrace"
            )
        new_seqs.append(index[key].seq)
    lo, hi = min(new_seqs), max(new_seqs)
    if sorted(new_seqs) != list(range(lo, hi + 1)):
        raise SweepDivergence(
            f"member ops are no longer consecutive in the retrace "
            f"(seqs {sorted(new_seqs)}); the stream diverged at this size"
        )
    return _boundary(retrace, workload, lo, hi)
