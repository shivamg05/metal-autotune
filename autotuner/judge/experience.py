"""A bounded view of experiments; complete, approved metadata stays readable."""
import copy

from .directions import DIRECTIONS, OPENERS

HISTORY_ROWS = 12
LESSON_ROWS = 6
INSPIRATION_CHARS = 2000


def excerpt(text, limit):
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def compact_experience(payload, archive):
    """archive registers read-only metadata through the existing source lookup."""
    meta = payload.get("region_state", payload)
    if "region" not in meta:
        return payload
    result = copy.deepcopy(payload)
    brief = result.get("region_state", result)
    brief["techniques_reference"] = archive({
        "examples": DIRECTIONS, "menu": brief.pop("menu", {}), "moves": brief.pop("moves", []),
    })
    kernels = meta.get("kernels", {})
    history = meta.get("history", [])
    lessons = meta.get("lessons", [])
    wanted = {meta.get(role) for role in ("head", "shipped", "scaffold")}
    # A transport/refusal verdict may have no kernel. Keep the last evaluated
    # candidate visible so the next successful reply can still repair it.
    for verdict in (payload.get("verdict"), meta.get("last_verdict")):
        if verdict:
            wanted.add(verdict.get("kernel_id"))
    brief["kernels"] = {kid: value for kid, value in brief.get("kernels", {}).items() if kid in wanted}

    # Keep distinct opening approaches, recent failures, current parents and recent work.
    selected, kinds = [], set()
    scaffold = meta.get("scaffold")
    for i, row in enumerate(history):
        if (scaffold is not None and row.get("kernel") and row.get("parent") == scaffold
                and row.get("kind") not in kinds
                and row.get("id") not in {"scaffold", "original", "scafix"}):
            selected.append(i)
            kinds.add(row.get("kind"))
            if len(kinds) == OPENERS:
                break
    if len(selected) == OPENERS:
        # Advance only for evaluated follow-ups; transport retries and lookups
        # must not silently rotate the alternative out of view.
        followups = sum(bool(row.get("kernel")) for row in history[selected[-1] + 1:])
        alternatives = [history[i] for i in selected
                        if history[i]["kernel"] not in wanted and history[i]["kernel"] in kernels]
        if alternatives:
            row = alternatives[followups % len(alternatives)]
            kid = row["kernel"]
            brief["inspiration"] = {
                "kernel_id": kid, "verdict": row.get("verdict"),
                "hypothesis": excerpt(row.get("hypothesis", ""), 180),
                "summary": excerpt(row.get("summary", ""), 240),
                "kernel": archive(kernels[kid]),
                "code_excerpts": code_excerpts(kernels[kid]),
            }
    selected.extend(i for i, row in enumerate(history) if row.get("kernel") in wanted - {None})
    failures = [i for i, row in enumerate(history) if row.get("verdict") in ("failed", "rolled_back")]
    selected.extend(failures[-2:])
    selected.extend(reversed(range(len(history))))
    selected = list(dict.fromkeys(selected))[:HISTORY_ROWS]
    brief["history"] = []
    for i in sorted(selected):
        row = history[i]
        summary = {k: row[k] for k in ("id", "kernel", "kind", "parent", "verdict",
                   "target_workload", "timing_case", "region_ms", "library_ms",
                   "win_ms", "sigma_ms", "failed_gate") if k in row}
        summary.update({k: excerpt(row[k], limit) for k, limit in (("hypothesis", 180), ("summary", 240)) if k in row})
        brief["history"].append(summary)

    ops = {op["op"] for op in meta["region"].get("ops", [])}
    relevant = [(i, lesson) for i, lesson in enumerate(lessons)
                if lesson.get("region") == meta["region"].get("fingerprint") or ops.intersection(lesson.get("ops", []))]
    relevant.sort(key=lambda pair: (pair[1].get("region") == meta["region"].get("fingerprint"),
                                   len(ops.intersection(pair[1].get("ops", []))), pair[0]), reverse=True)
    brief["lessons"] = [{"id": f"lesson_{i + 1}", "region": lesson.get("region"),
                          "lesson": excerpt(lesson["lesson"], 400),
                          "evidence": lesson.get("evidence", [])}
                         for i, lesson in relevant[:LESSON_ROWS]]
    brief["regions_done"] = brief.get("regions_done", [])[-4:]
    if history or lessons or len(kernels) > len(brief["kernels"]) or len(meta.get("regions_done", [])) > 4:
        brief["experience_archive"] = {
            **archive({"history": history, "kernels": kernels, "lessons": {f"lesson_{i + 1}": note for i, note in enumerate(lessons)},
                       "regions_done": meta.get("regions_done", [])}),
            "attempts": len(history), "lessons": len(lessons),
        }
    return result


def code_excerpts(kernel):
    """One small shared allowance, including headers and multi-stage kernels."""
    shown, remaining = [], INSPIRATION_CHARS

    def visit(value, path=""):
        nonlocal remaining
        if isinstance(value, dict):
            for key, child in value.items():
                location = f"{path}.{key}" if path else key
                if key in ("source", "header") and isinstance(child, str):
                    text = child[:remaining]
                    if text:
                        shown.append({"field": location, "start": 0, "text": text})
                        remaining -= len(text)
                elif isinstance(child, (dict, list, tuple)):
                    visit(child, location)
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                visit(child, f"{path}[{i}]")
    visit(kernel)
    return shown
