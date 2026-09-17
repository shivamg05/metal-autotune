"""Turn recorded region boundaries into portable graph rewrite rules.

The emitted class declares guards and patterns only. Its forward calculation
comes from the original module, not a generated replay of the module's body.
State handed to the scope as an argument compiles with its arrays as inputs
and outputs and its attributes in the call signature; state the scope reaches
any other way keeps replay delivery.
"""

import json

from autotuner_runtime.graph import GRAPH_CACHE_MAX

from ..trace.optable import MUTATING_METHODS
from ..trace.recorder import ObjectRef, state_method
from ..trace.serialize import _encode, node_to_dict
from ..trace.types import Retention
from .emit import EmittedWrapper, NotReplayable, _validate_splices, scope_nodes, scope_tree_path

_MUTATING = frozenset(f"array.{name}" for name in MUTATING_METHODS)


def graph_scope_reason(trace, scope):
    """Return why compiling this scope cannot preserve its Python behavior."""
    nodes = scope_nodes(trace, scope)
    if len(set(scope.obj_ids)) != len(scope.obj_ids):
        return "the scope aliases mutable Python state arguments"
    receivers = {node.scalar_args.get("receiver", {}).get("id"): node.scalar_args.get("receiver", {})
                 for node in reversed(nodes) if state_method(node.op)}
    for oid in scope.obj_ids:
        receiver = receivers.get(oid, {})
        if receiver.get("state_signature") is None:
            return "the scope accepts Python state without a complete tensor-state signature"
    if any(site[:len(scope.stack)] == scope.stack for site in trace.eval_sites):
        return "the scope evaluates arrays inside its Python call"
    # A compiled call computes every output together once one is needed: a
    # call mixing results the step uses with results it never uses would run
    # the latter here and nowhere else. A call nothing uses is dropped whole,
    # compiled or not.
    if any(node.seq in trace.dead for node in nodes) and any(node.seq not in trace.dead for node in nodes):
        return "the scope computes results the step never uses beside ones it does; compiling it would run them"
    # State reached through an explicit argument (a cache inside a cache
    # list) compiles with it; state reached any other way does not. A cache
    # position is part of the call signature: like a replay variant, the
    # compiled scope serves the positions it was recorded at.
    explicit = [receivers[oid].get("path") for oid in scope.obj_ids]
    produced = {a for node in nodes for a in node.out_arrays}
    for node in nodes:
        if state_method(node.op):
            receiver = node.scalar_args.get("receiver", {})
            path = receiver.get("path")
            if receiver.get("id") not in scope.obj_ids and not any(
                    root and path and path.startswith(root + ".") for root in explicit):
                return "the scope advances state outside its explicit arguments"
        # Compilation drops an in-place write to an array the caller still
        # holds; a write to the scope's own intermediate is ordinary dataflow.
        if node.op in _MUTATING and node.in_arrays[0] not in produced:
            return "the scope mutates an array it did not produce"
        if any(trace.liveness[a].kind is Retention.PYTHON_RETAINED for a in node.out_arrays):
            return "the scope retains intermediate arrays in Python state"
    return None


def _template(value):
    if isinstance(value, ObjectRef):
        return {"$": "object", "i": value.index}
    if isinstance(value, (tuple, list)):
        return {"$": type(value).__name__, "items": [_template(v) for v in value]}
    if isinstance(value, dict):
        return {"$": "dict", "items": {k: _template(v) for k, v in value.items()}}
    return _encode(value)


def _rule(trace, scope, splice):
    nodes = trace.nodes[splice.start_seq:splice.end_seq + 1]
    specs = trace.span_specs(splice.start_seq, splice.end_seq)
    # Normalize local array ids so equivalent copies share one match rule.
    names = {aid: i for i, aid in enumerate(splice.input_ids)}
    encoded = []
    for node in nodes:
        data = node_to_dict(node)
        for aid in node.out_arrays:
            names.setdefault(aid, len(names))
        data["in_arrays"] = [names[a] for a in node.in_arrays]
        data["out_arrays"] = [names[a] for a in node.out_arrays]
        for key in ("seq", "module_address", "position_in_module", "module_stack"):
            data.pop(key, None)
        encoded.append(data)
    path = scope_tree_path(scope.address)
    anchors = []
    for aid in splice.input_ids:
        if aid in scope.arg_ids:
            anchors.append(["argument", scope.arg_ids.index(aid)])
        elif aid in trace.weights:
            weight = trace.weight_paths.get(aid)
            if weight is None or (path and not weight.startswith(path + ".")):
                if scope.obj_ids:
                    anchors.append(None)  # tensor state supplied by an explicit object argument
                    continue
                raise NotReplayable("a graph boundary weight is outside its installation scope")
            anchors.append(["weight", weight[len(path):].lstrip(".")])
        else:
            anchors.append(None)
    return {"kernel_id": splice.kernel.kernel_id, "count": 1,
            "input_specs": [[list(specs[a][0]), specs[a][1]] for a in splice.input_ids],
            "anchors": anchors,
            "sequence": {"nodes": encoded, "input_ids": list(range(len(splice.input_ids))),
                         "output_ids": [names[a] for a in splice.output_ids]}}


def emit_graph_wrapper_variants(variants, class_name):
    if not variants:
        raise NotReplayable("a graph wrapper requires a recorded call")
    path = scope_tree_path(variants[0][1].address)
    records = []
    kernel_ids = []
    for trace, scope, splices in variants:
        reason = graph_scope_reason(trace, scope)
        if reason:
            raise NotReplayable(reason)
        if scope_tree_path(scope.address) != path:
            raise NotReplayable("graph variants must belong to one module")
        nodes = scope_nodes(trace, scope)
        if not nodes:
            raise NotReplayable("a graph scope must contain array operations")
        _validate_splices(nodes, splices, scope.address)
        specs = trace.span_specs(nodes[0].seq, nodes[-1].seq)
        rules = {}
        for splice in splices:
            rule = _rule(trace, scope, splice)
            key = json.dumps(rule, sort_keys=True)
            if key in rules:
                rules[key]["count"] += 1
            else:
                rules[key] = rule
            if splice.kernel.kernel_id not in kernel_ids:
                kernel_ids.append(splice.kernel.kernel_id)
        receivers = {node.scalar_args.get("receiver", {}).get("id"): node.scalar_args.get("receiver", {})
                     for node in reversed(nodes) if state_method(node.op)}
        record = {"args": _template(scope.args_template), "kwargs": _template(scope.kwargs_template),
                  "arg_specs": [[list(specs[a][0]), specs[a][1]] for a in scope.arg_ids],
                  "arg_aliases": [scope.arg_ids.index(a) for a in scope.arg_ids],
                  "weight_specs": {weight[len(path):].lstrip("."): [list(specs[aid][0]), specs[aid][1]]
                                   for aid, weight in trace.weight_paths.items()
                                   if aid in specs and (not path or weight.startswith(path + "."))},
                  "outputs": _encode(scope.out_template), "output_count": len(scope.out_ids),
                  "object_signatures": [json.dumps(receivers[oid]["state_signature"]) for oid in scope.obj_ids],
                  "rules": list(rules.values())}
        guard = (record["args"], record["kwargs"], record["arg_specs"], record["object_signatures"])
        existing = next((r for r in records
                         if (r["args"], r["kwargs"], r["arg_specs"], r["object_signatures"]) == guard), None)
        if existing is not None:
            if existing != record:
                raise NotReplayable("the same graph call signature selects different cuts")
        else:
            records.append(record)
    if len(records) > GRAPH_CACHE_MAX:
        raise NotReplayable(f"the scope has {len(records)} call signatures; a compiled scope keeps {GRAPH_CACHE_MAX}")
    source = ("\nimport json as _graph_json\nfrom autotuner_runtime.graph import GraphWrapper\n\n"
              f"class {class_name}(GraphWrapper):\n"
              f"    GRAPH_VARIANTS = _graph_json.loads({json.dumps(records)!r})\n"
              f"    KERNEL_IDS = {kernel_ids!r}\n"
              "    def __call__(self, *args, **kwargs):\n"
              "        return self._graph_call(args, kwargs)\n")
    return EmittedWrapper(class_name, path, source, [], kernel_ids)
