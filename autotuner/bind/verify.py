"""Literal retrace verification (plan 7.10). With replay-wrapper delivery the
spec's check is checkable directly: after a bind, a record-mode retrace of the
patched model must show the cut's member ops gone, one custom-kernel node per
copy consuming the cut's inputs and feeding its downstream consumers, and the
stream elsewhere unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..trace.types import Trace, TraceNode


@dataclass
class RetraceReport:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _project(nodes: list[TraceNode], spans: list[tuple[int, int]], kernel_ids: list[str]):
    """Collapse cut spans to custom-kernel events; keep (op, out_specs) and the
    original node (or span) per event for the dataflow check."""
    events = []
    i = 0
    span_idx = 0
    while i < len(nodes):
        if span_idx < len(spans) and i == spans[span_idx][0]:
            kid = kernel_ids[min(span_idx, len(kernel_ids) - 1)]
            events.append(("custom_kernel", kid, spans[span_idx]))
            i = spans[span_idx][1] + 1
            span_idx += 1
        else:
            n = nodes[i]
            events.append((n.op, n.out_specs, n))
            i += 1
    return events


def _consumer_events(trace: Trace, produced: set[int], event_seq_of: dict[int, int]) -> set[int]:
    """Projected event indices that read any of the produced arrays."""
    out = set()
    for node in trace.nodes:
        if any(a in produced for a in node.in_arrays):
            idx = event_seq_of.get(node.seq)
            if idx is not None:
                out.add(idx)
    return out


def verify_retrace(
    baseline: Trace,
    patched: Trace,
    cut_spans: list[tuple[int, int]],
    expected_kernel_ids: list[str],
) -> RetraceReport:
    reasons: list[str] = []
    spans = sorted(cut_spans)

    expected = _project(list(baseline.nodes), spans, expected_kernel_ids)

    got = []
    for n in patched.nodes:
        if n.op == "custom_kernel":
            got.append(("custom_kernel", n.scalar_args["kwargs"].get("kernel_id"), n))
        else:
            got.append((n.op, n.out_specs, n))

    if len(got) != len(expected):
        return RetraceReport(False, [
            f"patched stream has {len(got)} events, expected {len(expected)}: "
            f"the cut did not become one custom dispatch per copy"
        ])

    for k, (e, g) in enumerate(zip(expected, got)):
        if e[0] != g[0]:
            reasons.append(f"event {k}: expected op {e[0]!r}, retraced {g[0]!r}")
            break
        if e[0] == "custom_kernel":
            if e[1] != g[1]:
                reasons.append(f"event {k}: expected kernel {e[1]!r}, retraced {g[1]!r}")
                break
        elif e[1] != g[1]:
            reasons.append(
                f"event {k} ({e[0]!r}): output specs changed, {e[1]} -> {g[1]}: "
                f"neighbors are not unchanged"
            )
            break

    if not reasons:
        # dataflow: each custom node's outputs must feed the same downstream
        # events the baseline cut's outputs fed
        base_event_of = {}
        pat_event_of = {}
        for idx, (e, g) in enumerate(zip(expected, got)):
            if e[0] != "custom_kernel":
                base_event_of[e[2].seq] = idx
            pat_event_of[g[2].seq] = idx
        kernel_positions = [i for i, e in enumerate(expected) if e[0] == "custom_kernel"]
        for pos, span in zip(kernel_positions, spans):
            in_span = baseline.nodes[span[0]:span[1] + 1]
            produced = {a for n in in_span for a in n.out_arrays}
            base_consumers = _consumer_events(baseline, produced, base_event_of)
            pat_node = got[pos][2]
            pat_consumers = _consumer_events(patched, set(pat_node.out_arrays), pat_event_of)
            if base_consumers != pat_consumers:
                reasons.append(
                    f"custom dispatch at event {pos}: outputs feed events "
                    f"{sorted(pat_consumers)}, baseline cut fed {sorted(base_consumers)}"
                )

    return RetraceReport(ok=not reasons, reasons=reasons)
