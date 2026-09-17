"""Focused source excerpts and read-only lookup over already-approved metadata.

Offsets address Python string characters, never files. Excerpts are navigation
hints, not a C++ parser or an executable rewrite. Original code stays untouched.
"""
import re

from .schema import MalformedResponse


INLINE_CHARS = 8000
READ_CHARS = 8000
PREVIEW_CHARS = 12000  # shared across the briefing, not multiplied by history
MAX_READ_ROUNDS = 8
MAX_READS = 4


class SourceContext:
    def __init__(self):
        self.sources = {}
        self._ids = {}

    def focus(self, payload):
        catalog = {}

        def visit(value, field=None, body=""):
            if field in ("source", "header") and isinstance(value, str) and len(value) > INLINE_CHARS:
                if value not in self._ids:
                    source_id = f"source_{len(self.sources) + 1}"
                    self._ids[value] = source_id
                    self.sources[source_id] = value
                    catalog[source_id] = {
                        "characters": len(value), "lines": value.count("\n") + 1,
                        "excerpts": [],
                    }
                source_id = self._ids[value]
                entry = catalog[source_id]
                # Look for the names called by the body. This deliberately does
                # not claim to resolve overloads, macros or transitive helpers.
                calls = re.findall(r"\b([A-Za-z_]\w{2,})\s*(?:<[^;{}()]{1,500}>)?\s*\(", body)
                starts = []
                if field == "header":
                    for name in dict.fromkeys(calls):
                        if name in {"sizeof", "alignof", "decltype", "static_cast", "reinterpret_cast"}:
                            continue
                        match = re.search(r"\b" + re.escape(name) + r"\s*\(", value)
                        if match:
                            starts.append(max(0, match.start() - 400))
                            if len(starts) == 2:
                                break
                for start in starts[:2] or [0]:
                    if len(entry["excerpts"]) >= 2:
                        break
                    if any(abs(excerpt["start"] - start) < 3000 for excerpt in entry["excerpts"]):
                        continue
                    entry["excerpts"].append(self._slice(source_id, start, 6000))
                return {"source_id": source_id}
            if isinstance(value, dict):
                body = value.get("source", "")
                body = body if isinstance(body, str) else ""
                return {key: visit(child, key, body) for key, child in value.items()}
            if isinstance(value, (list, tuple)):
                return [visit(child) for child in value]
            return value

        result = visit(payload)
        if catalog:
            # Spend preview space on current work first. Other versions remain
            # fully readable; their number must not multiply the opening brief.
            meta = result.get("region_state", result)
            latest = payload.get("verdict") or {}
            latest = latest if isinstance(latest, dict) else {}
            preferred = []

            def source_ids(value):
                if isinstance(value, dict):
                    if set(value) == {"source_id"} and isinstance(value["source_id"], str) and value["source_id"] in catalog:
                        preferred.append(value["source_id"])
                    else:
                        for child in value.values():
                            source_ids(child)
                elif isinstance(value, list):
                    for child in value:
                        source_ids(child)

            for kernel_id in (meta.get("head"), latest.get("kernel_id"), meta.get("shipped")):
                source_ids(meta.get("kernels", {}).get(kernel_id, {}))
            remaining = PREVIEW_CHARS
            for source_id in dict.fromkeys(preferred + list(catalog)):
                entry = catalog[source_id]
                shown = []
                for excerpt in entry["excerpts"]:
                    if remaining <= 0:
                        break
                    shown.append(self._slice(source_id, excerpt["start"], min(remaining, len(excerpt["text"]))))
                    remaining -= len(shown[-1]["text"])
                entry["excerpts"] = shown
            result["source_catalog"] = catalog
        return result

    def _slice(self, source_id, start, length):
        source = self.sources[source_id]
        end = min(len(source), start + length)
        return {"start": start, "end": end, "text": source[start:end],
                "next_start": end if end < len(source) else None}

    def read(self, obj):
        """Validate an entire batch before returning any text. No path/regex API."""
        requests = obj.get("read_source")
        if set(obj) != {"read_source"} or not isinstance(requests, list) or not 1 <= len(requests) <= MAX_READS:
            raise MalformedResponse(f"read_source must be the only key, with 1-{MAX_READS} requests")
        results = []
        for request in requests:
            if not isinstance(request, dict) or set(request) - {"id", "start", "length", "find"}:
                raise MalformedResponse("source request keys: id, optional start, and length or find")
            source_id = request.get("id")
            if not isinstance(source_id, str) or source_id not in self.sources:
                raise MalformedResponse("unknown source id; use an id in source_catalog")
            source = self.sources[source_id]
            start = request.get("start", 0)
            if type(start) is not int or not 0 <= start <= len(source):
                raise MalformedResponse("source start must be a character offset within the source")
            if "find" in request:
                needle = request["find"]
                if "length" in request or not isinstance(needle, str) or not 1 <= len(needle) <= 200:
                    raise MalformedResponse("find must be 1-200 literal characters; omit length")
                matches = []
                offset = source.find(needle, start)
                while offset >= 0 and len(matches) < 20:
                    matches.append({"offset": offset,
                                    "excerpt": self._slice(source_id, max(0, offset - 100), len(needle) + 200)})
                    offset = source.find(needle, offset + len(needle))
                results.append({"id": source_id, "matches": matches,
                                "next_start": offset if offset >= 0 else None})
            else:
                length = request.get("length", READ_CHARS)
                if type(length) is not int or not 1 <= length <= READ_CHARS:
                    raise MalformedResponse(f"source length must be 1-{READ_CHARS} characters")
                results.append({"id": source_id, **self._slice(source_id, start, length)})
        return results
