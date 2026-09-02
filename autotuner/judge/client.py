"""The Anthropic API client behind the judge interface.

Strict JSON in and out: every response goes through schema.validate_response;
a malformed one gets exactly one re-ask carrying the rejection reason, and a
second miss raises JudgeBabble, which the loop maps to a failed hypothesis
(budget burn, never the compile-fail streak). The judge is stateless per
call: each exchange is one fresh conversation over the rendered region state.

Model is configurable per job. Current top-tier Anthropic models reject
sampling parameters (temperature and friends return 400), so none are sent
unless explicitly configured for an older model; reproducibility comes from
the strict schema, the re-ask, and the run log, not from sampling.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..log import json_safe, wall_now
from .examples import render_examples
from .schema import (
    JudgeBabble,
    MalformedResponse,
    NextResponse,
    SeedResponse,
    validate_response,
)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000

_SYSTEM = """You are the judge in an autonomous optimizer for MLX models on Apple \
Silicon. A harness recorded the ops a model runs, cut out one region (a run of \
ops that a single Metal kernel could replace), built a correct starting kernel \
for it, and now asks you for one small edit at a time. You see the region's \
shapes and costs, the kernels in play with their verdicts, and a legend for \
every field; you never see tensors, weights, activations, or tolerance values. \
The harness compiles, checks, and times every kernel you write; its verdicts \
are final, and it owns the call site: kernel names, init_value, math_mode, \
and streams are not yours to set.

You plan a queue of hypotheses in English and write Metal only for the item \
named in writing_for, as an edit of a named parent kernel. Each hypothesis \
carries a kind, a short label in your own words for the move it makes. The \
menu lists common kinds and moves lists where wins tend to come from; neither \
is a limit. Anything the laws allow is fair game, and what you see in the \
region and its verdicts is yours to act on, including ideas that turn out \
slower: a correct kernel is never wasted, it becomes a parent. Respond with \
exactly one JSON object and nothing else: no prose, no code fences, no keys \
beyond the schema.

{schema}"""

_ITEM_SCHEMA = """item: {"id": str, "kind": short label in your own words (menu lists common ones),
       "assoc_tag": "preserving"|"changing", "hypothesis": English string,
       optional "family_id": str,
       optional "depends_on": an earlier item's id, with "condition": "correct"|"shipped"|"failed"}
An id is letters, digits, and underscores only: it becomes part of a kernel name.
No other item keys exist."""

_LESSON = """optional "lesson": one sentence, under 400 characters, that this region taught and
later regions of this job should know (a numerics rule a gate enforced, a layout that
paid); it is shown with every later call as lessons."""

_SEED_SCHEMA = f"""Response schema (seed):
{{"queue": [item, ...], {_LESSON}}}
{_ITEM_SCHEMA}"""

_NEXT_SCHEMA = f"""Response schema (next):
{{"mutations": [mutation, ...], "kernel": proposal or null, {_LESSON}}}
mutation: {{"op": "insert", "item": item, optional "before": queued id}}
        | {{"op": "delete", "id": queued id}}
        | {{"op": "reorder", "order": [every queued id, new order]}}
{_ITEM_SCHEMA}
""" + """proposal, the Metal for the item in writing_for, an edit of a named parent:
  {"source": kernel body, "parent_kernel_id": str, optional "item_id": str,
   optional "header": str,
   "grid": [3 launch-grammar exprs, total threads], "threadgroup": [3 exprs],
   "output_shapes": [[exprs], one entry per region output, in io.outputs order],
   optional "scratch": [[name, dtype name, [shape exprs]], ...],
   optional "template": [[name, dtype name or "inN"], ...],
   optional "fallback_predicate": launch-grammar predicate}
parent_kernel_id names the kernel you edited: head, scaffold, shipped, a hypothesis
id, or a kernel id from kernels. source is the whole kernel body; a header or
template you leave out is inherited from the parent, so omit them to keep the
parent's. Extra device buffers the body
writes (staging between stages) go in scratch, named tmp0, tmp1, ... in the order
the body uses them; region outputs are always out0, out1, ... and inputs in0, in1, ...
The verdict you are sent is for the last kernel you wrote; mutate the queue in
reply to it first (insert a fix, drop a dead family, reorder), then write for
the item that is front and ready after those mutations. item_id names it when
it is not the item in writing_for.
A yield ("kernel": null) is refused while the region's budget lasts: the harness
asks again with the reason in the verdict as plan_refused, and after that one free
re-ask every reply with nothing to evaluate costs an attempt, the same as a plan
edit the queue refuses or a kernel for an item that is not ready. budget in
region_state says how many attempts remain; spend them all: a correct kernel is
never wasted, and a different family is always worth a try."""


class JsonJudge:
    """The exchange loop every transport shares: strict schema, one re-ask
    carrying the rejection reason, then JudgeBabble. Subclasses supply
    _ask(system, messages) -> the judge's raw reply text."""

    transcript: Path | None = None  # every ask and reply, one JSON line each

    def seed(self, region_meta: dict) -> SeedResponse:
        return self._exchange({"region_state": region_meta},
                              _SEED_SCHEMA, SeedResponse)

    def next(self, region_meta: dict, verdict: object) -> NextResponse:
        return self._exchange({"region_state": region_meta, "verdict": verdict},
                              _NEXT_SCHEMA, NextResponse)

    def _exchange(self, payload: dict, schema_doc: str, want: type):
        system = _SYSTEM.format(schema=schema_doc) + "\n\n" + render_examples(
            "seed" if want is SeedResponse else "next")
        messages = [{"role": "user", "content": json.dumps(payload)}]
        reason = None
        for _attempt in range(2):  # the original ask, then the one re-ask
            text = self._ask(system, messages)
            self._record(want.__name__, _attempt, system, messages, text)
            try:
                obj = json.loads(text)
            except (json.JSONDecodeError, RecursionError) as e:
                # RecursionError: pathologically nested JSON exhausts the parser
                reason = MalformedResponse(f"response is not parseable JSON: {type(e).__name__}")
            else:
                try:
                    response = validate_response(obj)
                    if not isinstance(response, want):
                        raise MalformedResponse(f"expected a {want.__name__} shape")
                    return response
                except MalformedResponse as e:
                    reason = e
            messages = messages + [
                {"role": "assistant", "content": text or "(empty)"},
                {"role": "user", "content":
                    f"Your response was rejected: {reason}. Respond again with "
                    "exactly one JSON object matching the schema, nothing else."},
            ]
        raise JudgeBabble(str(reason))

    def _ask(self, system: str, messages: list[dict]) -> str:
        raise NotImplementedError

    def _record(self, call: str, attempt: int, system: str, messages: list[dict], reply: str) -> None:
        if self.transcript is None:
            return
        row = {"wall": wall_now(), "call": call, "attempt": attempt,
               "system": system, "messages": messages, "reply": reply}
        self.transcript.parent.mkdir(parents=True, exist_ok=True)
        with self.transcript.open("a") as f:
            f.write(json.dumps(json_safe(row), allow_nan=False) + "\n")


class AnthropicJudge(JsonJudge):
    """Same two entry points as the scripted judge, over the live API."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float | None = None, client=None):
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = client  # injectable transport for tests; SDK built lazily

    def _ask(self, system: str, messages: list[dict]) -> str:
        kwargs = {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        response = self._sdk().messages.create(
            model=self._model, max_tokens=self._max_tokens,
            system=system, messages=messages, **kwargs,
        )
        if getattr(response, "stop_reason", None) == "refusal":
            return ""
        return "".join(b.text for b in response.content if b.type == "text")

    def _sdk(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client
