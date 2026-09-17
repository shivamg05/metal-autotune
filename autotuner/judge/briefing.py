"""Lossless sharing of repeated code and evidence inside one fresh briefing.

No history is dropped and no state is assumed from an earlier request.
Only large, identical values are shared; small fields stay next to their
kernel so reading the briefing does not become a reference hunt.
"""
from collections import Counter
import copy
import json


_SHARED_FIELDS = {"source", "header", "model_check", "incumbent_screen"}
_MIN_SHARED_CHARS = 256  # below this, a table entry and references buy little


def _key(field, value):
    if field not in _SHARED_FIELDS or not isinstance(value, (str, dict)) or not value:
        return None
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False)
    return encoded if len(encoded) >= _MIN_SHARED_CHARS else None


def share_context(payload: dict) -> dict:
    """Replace repeated large values with {context_ref: name}, preserving
    their exact contents in shared_context. Leaves the input untouched.
    References are a display format, never executable proposal fields.
    """
    def contains_marker(value):
        if isinstance(value, dict):
            return "context_ref" in value or "shared_context" in value or any(
                contains_marker(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_marker(child) for child in value)
        return False

    # Arbitrary model metadata may itself use these names. Keep that request
    # inline rather than making its literal values look like our references.
    if contains_marker(payload):
        return copy.deepcopy(payload)
    counts = Counter()

    def count(value, field=None):
        key = _key(field, value)
        if key is not None:
            counts[key] += 1
            return
        if isinstance(value, dict):
            for field, child in value.items():
                count(child, field)
        elif isinstance(value, (list, tuple)):
            for child in value:
                count(child)

    count(payload)
    names, shared = {}, {}

    def render(value, field=None):
        key = _key(field, value)
        if key is not None and counts[key] > 1:
            if key not in names:
                name = f"context_{len(names) + 1}"
                names[key] = name
                shared[name] = json.loads(key)
            return {"context_ref": names[key]}
        if isinstance(value, dict):
            return {field: render(child, field) for field, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [render(child) for child in value]
        return value

    result = render(payload)
    if shared:
        result["shared_context"] = shared
    return result
