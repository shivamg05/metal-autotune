"""The Anthropic API client behind the judge interface (plan 5.11).

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

from .schema import (
    JudgeBabble,
    MalformedResponse,
    NextResponse,
    SeedResponse,
    validate_response,
)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000

_SYSTEM = """You are the judge in an autonomous Metal kernel optimizer for MLX \
on Apple Silicon. You see region metadata only, never tensors, weights, \
activations, or tolerance values. You plan a hypothesis queue in English and \
write Metal only for the front ready item, one small edit of a named parent \
at a time. The harness owns the kernel call site: kernel names, init_value, \
math_mode, and streams are not yours to set, and a failed check is final.

Respond with exactly one JSON object and nothing else: no prose, no code \
fences, no keys beyond the schema.

{schema}"""

_ITEM_SCHEMA = """item: {"id": str, "kind": "on-chip"|"specialize"|"retile"|"re-layout"|"algorithm"|"launch"|"fix",
       "assoc_tag": "preserving"|"changing", "hypothesis": English string,
       optional "family_id": str,
       optional "depends_on": an earlier item's id, with "condition": "correct"|"shipped"|"failed"}
No other item keys exist."""

_SEED_SCHEMA = f"""Response schema (seed):
{{"queue": [item, ...]}}
{_ITEM_SCHEMA}"""

_NEXT_SCHEMA = f"""Response schema (next):
{{"mutations": [mutation, ...], "kernel": proposal or null}}
mutation: {{"op": "insert", "item": item, optional "before": queued id}}
        | {{"op": "delete", "id": queued id}}
        | {{"op": "reorder", "order": [every queued id, new order]}}
{_ITEM_SCHEMA}
""" + """proposal, the Metal for the front ready item, an edit of a named parent:
  {"source": kernel body, "parent_kernel_id": str, optional "header": str,
   "grid": [3 launch-grammar exprs, total threads], "threadgroup": [3 exprs],
   "output_shapes": [[exprs] per output],
   optional "template": [[name, dtype name or "inN"], ...],
   optional "fallback_predicate": launch-grammar predicate}
Return "kernel": null to yield when you have nothing left to propose."""


class JsonJudge:
    """The exchange loop every transport shares: strict schema, one re-ask
    carrying the rejection reason, then JudgeBabble. Subclasses supply
    _ask(system, messages) -> the judge's raw reply text."""

    def seed(self, region_meta: dict) -> SeedResponse:
        return self._exchange({"region_state": region_meta},
                              _SEED_SCHEMA, SeedResponse)

    def next(self, region_meta: dict, verdict: object) -> NextResponse:
        return self._exchange({"region_state": region_meta, "verdict": verdict},
                              _NEXT_SCHEMA, NextResponse)

    def _exchange(self, payload: dict, schema_doc: str, want: type):
        system = _SYSTEM.format(schema=schema_doc)
        messages = [{"role": "user", "content": json.dumps(payload)}]
        reason = None
        for _attempt in range(2):  # the original ask, then the one re-ask
            text = self._ask(system, messages)
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
