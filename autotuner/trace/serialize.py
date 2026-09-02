"""TraceNode serialization: nodes cross the sandbox boundary as JSON so a
patch-free worker can replay a region span from the spec alone: nothing is
pickled, no state is shared."""

from __future__ import annotations

import json
from typing import Any

import mlx.core as mx

from .recorder import ArrayRef, dtype_name
from .types import TraceNode


def _encode(v: Any) -> Any:
    if isinstance(v, ArrayRef):
        return {"$": "ref", "i": v.index}
    if isinstance(v, tuple):
        return {"$": "tuple", "items": [_encode(x) for x in v]}
    if isinstance(v, list):
        return {"$": "list", "items": [_encode(x) for x in v]}
    if isinstance(v, dict):
        return {"$": "dict", "items": {k: _encode(x) for k, x in v.items()}}
    if isinstance(v, slice):
        return {"$": "slice", "parts": [_encode(v.start), _encode(v.stop), _encode(v.step)]}
    if isinstance(v, mx.Dtype):
        return {"$": "dtype", "name": dtype_name(v)}
    if v is Ellipsis:
        return {"$": "ellipsis"}
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    raise TypeError(f"cannot serialize scalar arg of type {type(v).__name__}: {v!r}")


def _decode(v: Any) -> Any:
    if isinstance(v, dict) and "$" in v:
        tag = v["$"]
        if tag == "ref":
            return ArrayRef(v["i"])
        if tag == "tuple":
            return tuple(_decode(x) for x in v["items"])
        if tag == "list":
            return [_decode(x) for x in v["items"]]
        if tag == "dict":
            return {k: _decode(x) for k, x in v["items"].items()}
        if tag == "slice":
            return slice(*(_decode(x) for x in v["parts"]))
        if tag == "dtype":
            return getattr(mx, v["name"] if v["name"] != "bool" else "bool_")
        if tag == "ellipsis":
            return Ellipsis
        raise ValueError(f"unknown tag {tag!r}")
    return v


def node_to_dict(node: TraceNode) -> dict:
    return {
        "seq": node.seq,
        "op": node.op,
        "in_arrays": list(node.in_arrays),
        "out_arrays": list(node.out_arrays),
        "in_specs": [[list(s), d] for s, d in node.in_specs],
        "out_specs": [[list(s), d] for s, d in node.out_specs],
        "args": _encode(tuple(node.scalar_args["args"])),
        "kwargs": {k: _encode(v) for k, v in node.scalar_args["kwargs"].items()},
        "module_address": node.module_address,
        "position_in_module": node.position_in_module,
        "module_stack": list(node.module_stack),
    }


def node_from_dict(d: dict) -> TraceNode:
    return TraceNode(
        seq=d["seq"],
        op=d["op"],
        in_arrays=tuple(d["in_arrays"]),
        out_arrays=tuple(d["out_arrays"]),
        in_specs=tuple((tuple(s), dt) for s, dt in d["in_specs"]),
        out_specs=tuple((tuple(s), dt) for s, dt in d["out_specs"]),
        scalar_args={"args": _decode(d["args"]), "kwargs": {k: _decode(v) for k, v in d["kwargs"].items()}},
        module_address=d["module_address"],
        position_in_module=d["position_in_module"],
        module_stack=tuple(d["module_stack"]),
    )


def nodes_to_json(nodes) -> str:
    return json.dumps([node_to_dict(n) for n in nodes])


def nodes_from_json(text: str) -> list[TraceNode]:
    return [node_from_dict(d) for d in json.loads(text)]
