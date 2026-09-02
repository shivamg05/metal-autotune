"""Worked examples for the judge: an abridged region state, then a reply the
validator accepts. Rendered into the system prompt of every call, seed
examples for seed calls and next examples for next calls. The kernel in the
first next example is the one tests/test_loop.py ships on the planted-win
fixture, so an example never teaches a kernel the harness would reject."""

from __future__ import annotations

import json

# Where wins come from inside one region, in plain words, for the judge.
MOVES = (
    "read every input byte once and write every output byte once: fuse producer "
    "into consumer, keep intermediates in registers or threadgroup memory",
    "do no redundant work: a conversion, repack, or re-normalization the recorded "
    "ops perform twice can be done once",
    "fold the small ops after a matmul (bias, norm, rope, residual, activation) "
    "into the matmul's write-out instead of a separate pass over the output",
    "choose tiles, work per thread, and loop order for these exact shapes, with a "
    "fallback_predicate for every shape the specialization does not cover",
    "split a long reduction across SIMD groups and combine partials in "
    "threadgroup memory; this reorders the arithmetic, so tag it changing",
    "fewer launches matter only when bound is launch; when bound is memory, bytes "
    "moved is the whole story and the roofline says how far there is to go",
)

FUSED_CHAIN_SOURCE = """\
uint i = thread_position_in_grid.x;
uint c = i % (uint)in0_shape[1];
float x = in0[i];
float y = x * 2.0f + in1[c];
y = (y != y) ? y : metal::max(y, 0.0f);
y = y * x;
y = y + in2[c];
y = (y != y) ? y : metal::min(y, 8.0f);
out0[i] = (y - 1.0f) * 0.5f;
"""

_ROW_SUM_BROKEN = """\
uint r = thread_position_in_grid.x;
uint n = (uint)in0_shape[1];
for (uint c = 0; c < n; ++c) { acc += in0[r * n + c]; }
out0[r] = acc;
"""

_ROW_SUM_FIXED = """\
uint r = thread_position_in_grid.x;
uint n = (uint)in0_shape[1];
float acc = 0.0f;
for (uint c = 0; c < n; ++c) { acc += in0[r * n + c]; }
out0[r] = acc;
"""

SEED_EXAMPLES = [
    {
        "title": "seed: a norm feeding a projection, bound by memory",
        "state": {
            "region": {
                "ops": ["mx.fast.rms_norm", "array.__matmul__"],
                "io": {"decode": {"inputs": [[[1, 1024], "float16"], [[1024], "float16"],
                                             [[1024, 3072], "float16"]],
                                  "outputs": [[[1, 3072], "float16"]]}},
                "copies": 28, "bound": "memory", "T_rep_ms": 0.081, "roofline_ms": 0.062,
                "s_max": 1.31, "head_ms": 0.35, "shipped_ms": None,
            },
            "head": "scaffold",
            "note": "the scaffold normalizes the row into a scratch buffer, then a "
                    "second stage reads it back for every column of the weight",
        },
        "reply": {"queue": [
            {"id": "h1", "kind": "one-pass", "assoc_tag": "preserving", "family_id": "stream_w",
             "hypothesis": "normalize the row once into threadgroup memory, then every thread "
                           "streams its columns of the weight with 16-byte loads, so each weight "
                           "byte is read exactly once and nothing goes back to device memory"},
            {"id": "h1_fix", "kind": "fix", "assoc_tag": "preserving",
             "hypothesis": "repair whatever h1 fails on", "depends_on": "h1", "condition": "failed"},
            {"id": "h2", "kind": "split-K across simdgroups", "assoc_tag": "changing",
             "family_id": "stream_w", "depends_on": "h1", "condition": "correct",
             "hypothesis": "each SIMD group owns a K slice of every dot product and the partials "
                           "tree-reduce in threadgroup memory; the accumulation order changes"},
            {"id": "h3", "kind": "launch", "assoc_tag": "preserving", "family_id": "stream_w",
             "depends_on": "h1", "condition": "shipped",
             "hypothesis": "eight output columns per thread instead of four, halving the grid"},
        ]},
    },
]

NEXT_EXAMPLES = [
    {
        "title": "next: write the front item as an edit of the scaffold",
        "state": {
            "region": {
                "ops": ["array.__mul__", "array.__add__", "mx.maximum", "array.__mul__",
                        "array.__add__", "mx.minimum", "array.__sub__", "array.__mul__"],
                "io": {"main": {"inputs": [[[4096, 1024], "float32"], [[1024], "float32"],
                                           [[1024], "float32"]],
                                "outputs": [[[4096, 1024], "float32"]]}},
                "bound": "memory", "s_max": 6.2,
            },
            "head": "scaffold",
            "verdict": {"kernel": "scaffold", "outcome": "correct_slower", "region_ms": 18.2,
                        "library_ms": 12.5},
            "writing_for": {"id": "h1", "kind": "one-pass",
                            "hypothesis": "one thread per element carries the chain in a register"},
            "note": "the scaffold ran the eight ops as eight stages through scratch buffers; the "
                    "scalars are x*2, +in1, max 0, *x, +in2, min 8, -1, *0.5, and mx.maximum "
                    "and mx.minimum keep NaN",
        },
        "reply": {"mutations": [], "kernel": {
            "source": FUSED_CHAIN_SOURCE, "parent_kernel_id": "scaffold",
            "grid": ["in0.shape[0] * in0.shape[1]", "1", "1"],
            "threadgroup": ["min(in0.shape[0] * in0.shape[1], 256)", "1", "1"],
            "output_shapes": [["in0.shape[0]", "in0.shape[1]"]],
        }},
    },
    {
        "title": "next: the last kernel failed to compile, so insert a fix and write it",
        "state": {
            "region": {"ops": ["mx.sum"], "io": {"main": {"inputs": [[[4096, 1024], "float32"]],
                                                         "outputs": [[[4096], "float32"]]}}},
            "verdict": {"kernel": "h1", "outcome": "failed", "gate": "compile",
                        "detail": {"diagnostics": [
                            {"line": 3, "message": "use of undeclared identifier 'acc'"}]}},
            "queue": [{"id": "h2", "kind": "vectorize", "depends_on": "h1", "condition": "correct"}],
            "writing_for": None,
            "kernels": {"h1": {"source": _ROW_SUM_BROKEN}},
        },
        "reply": {
            "mutations": [{"op": "insert", "before": "h2", "item": {
                "id": "h1_fix", "kind": "fix", "assoc_tag": "preserving",
                "hypothesis": "declare the accumulator before the loop",
                "depends_on": "h1", "condition": "failed"}}],
            "kernel": {
                "source": _ROW_SUM_FIXED, "parent_kernel_id": "h1", "item_id": "h1_fix",
                "grid": ["in0.shape[0]", "1", "1"],
                "threadgroup": ["min(in0.shape[0], 256)", "1", "1"],
                "output_shapes": [["in0.shape[0]"]],
            },
        },
    },
    {
        "title": "next: a family that keeps losing is dropped, and nothing else is worth writing",
        "state": {
            "verdict": {"kernel": "h3", "outcome": "correct_slower", "region_ms": 0.093,
                        "library_ms": 0.081, "shipped_ms": 0.070},
            "queue": [{"id": "h4", "kind": "launch", "family_id": "stream_w",
                       "hypothesis": "sixteen columns per thread"}],
            "note": "h2 and h3 both lost to the shipped kernel; the family's next step "
                    "would move the same knob again",
        },
        "reply": {"mutations": [{"op": "delete", "id": "h4"}], "kernel": None},
    },
]


def render_examples(call: str) -> str:
    """The examples for one call kind ("seed" or "next") as prompt text."""
    examples = SEED_EXAMPLES if call == "seed" else NEXT_EXAMPLES
    parts = ["Worked examples. Each shows an abridged region state, then a reply the "
             "harness accepts. Kinds are the judge's own words."]
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i} ({ex['title']}):\nstate: {json.dumps(ex['state'])}\n"
                     f"reply: {json.dumps(ex['reply'])}")
    return "\n\n".join(parts)
