"""Compile the original module with certified graph substitutions.

Python constructs the original calculation when MLX traces a new call
signature; the native rewriter changes the selected subgraphs before
compilation. Warm calls execute that compiled graph. The module's arrays are
the compiled function's captured state, so MLX reads the current weights on
every call and re-traces the scope when one changes shape or dtype.

Python state handed to the scope (a cache) is explicit: its arrays enter the
compiled call as inputs and leave as outputs, and its plain attributes (an
offset) are part of the call signature. A warm call writes the arrays and
attributes the traced call produced back into the same objects, so a cache
advances exactly as the original Python advanced it. A signature never seen,
an offset included, runs the original module untouched.

A warm call is a signature, a cache lookup and one compiled call; everything
that decides which variant applies runs once per new key. The compiled call
itself still costs host time (about 10 to 20 us per scope on Qwen3 0.6B,
within the paired clock's noise there). Python settings on the module are
read when a scope's graph is built, exactly as a replay wrapper reads them
when it is generated; a scope whose settings change between calls is
unsupported.
"""

from contextlib import contextmanager
import copy
import json

import mlx.core as mx

from . import kernels
from .original_sequence import prepare_sequence
from .swap import ReplayWrapper, flatten_arrays, resolve_value, state_signature


class GraphBindingError(RuntimeError):
    """The original graph no longer contains exactly the certified cuts."""


class _PythonStateChanged(GraphBindingError):
    """The scope moved Python state during its call; only replay can carry it."""


class _GraphCache(dict):
    """Compiled closures are implementation details, never model state."""

    _trace_internal = True


class _Slot:
    """Stands in for one state array inside a frozen copy of a holder."""

    __slots__ = ("index",)

    def __init__(self, index):
        self.index = index


def execute_graph(function, arrays, state):
    """One traceable boundary for the already transformed computation."""
    return function(arrays, state)


GRAPH_CACHE_MAX = 64  # compiled call signatures kept per scope; each recorded cache position is one


def _python_snapshot(root):
    """Shallow snapshots preserve live modules and array handles on rollback."""
    saved, seen = [], set()

    def visit(value):
        if id(value) in seen or isinstance(value, mx.array):
            return
        seen.add(id(value))
        if isinstance(value, dict):
            saved.append((value, dict(value)))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            saved.append((value, list(value)))
            for child in value:
                visit(child)
        elif isinstance(value, set):
            saved.append((value, set(value)))
        elif isinstance(value, tuple):
            for child in value:
                visit(child)
        if hasattr(value, "__dict__") and (isinstance(value, dict) or not callable(value)):
            visit(vars(value))

    visit(root)
    return saved


def _rewind_python(saved):
    for target, original in reversed(saved):
        if isinstance(target, dict):
            dict.clear(target)
            dict.update(target, original)
        elif isinstance(target, set):
            set.clear(target)
            set.update(target, original)
        else:
            target[:] = original


def _configuration(value, seen=None):
    """Python choices belong in the guard; tensor contents remain live inputs."""
    if isinstance(value, mx.array):
        return "array"
    if value is None or isinstance(value, (int, float, str, bool, mx.Dtype)):
        return value
    seen = {} if seen is None else seen
    if id(value) in seen:
        return ("alias", seen[id(value)])
    seen[id(value)] = len(seen)
    if isinstance(value, dict):
        return (type(value), tuple((k, _configuration(v, seen)) for k, v in value.items()),
                _configuration(vars(value), seen) if hasattr(value, "__dict__") else ())
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_configuration(v, seen) for v in value))
    if isinstance(value, set):
        return (set, frozenset(value))
    if hasattr(value, "__dict__") and not callable(value):
        return (type(value), _configuration(vars(value), seen))
    return (type(value), id(value))


def _restore(template, arrays, objects=()):
    if isinstance(template, dict) and "$" in template:
        tag = template["$"]
        if tag == "ref":
            return arrays[template["i"]]
        if tag == "object":
            return objects[template["i"]]
        if tag in ("tuple", "list"):
            values = [_restore(x, arrays, objects) for x in template["items"]]
            return tuple(values) if tag == "tuple" else values
        if tag == "dict":
            return {key: _restore(x, arrays, objects) for key, x in template["items"].items()}
        if tag == "dtype":
            return getattr(mx, "bool_" if template["name"] == "bool" else template["name"])
        if tag == "slice":
            return slice(*[_restore(x, arrays) for x in template["parts"]])
        if tag == "ellipsis":
            return Ellipsis
        raise GraphBindingError(f"unsupported call template {tag!r}")
    return template


def _matches(template, value, specs):
    if isinstance(template, dict) and "$" in template:
        tag = template["$"]
        if tag == "ref":
            shape, dtype = specs[template["i"]]
            return (isinstance(value, mx.array) and tuple(value.shape) == tuple(shape)
                    and str(value.dtype).removeprefix("mlx.core.") == dtype)
        if tag == "object":
            return hasattr(value, "__dict__")
        if tag in ("tuple", "list"):
            expected = tuple if tag == "tuple" else list
            return (isinstance(value, expected) and len(value) == len(template["items"])
                    and all(_matches(t, v, specs) for t, v in zip(template["items"], value)))
        if tag == "dict":
            # Order matters: array references bind by flattening position.
            return (isinstance(value, dict) and list(value) == list(template["items"])
                    and all(_matches(t, value[k], specs) for k, t in template["items"].items()))
    try:
        return type(value) is type(_restore(template, [])) and value == _restore(template, [])
    except (TypeError, ValueError):
        return False


def _objects(template, value, result):
    if not isinstance(template, dict) or "$" not in template:
        return
    tag = template["$"]
    if tag == "object":
        result[template["i"]] = value
    elif tag in ("tuple", "list"):
        for t, v in zip(template["items"], value):
            _objects(t, v, result)
    elif tag == "dict":
        for key, t in template["items"].items():
            _objects(t, value[key], result)


def _is_holder(value):
    """A plain object carrying state in its attributes, the way a cache does."""
    return hasattr(value, "__dict__") and not callable(value) and not isinstance(value, mx.array)


def _signature(value, holders):
    """A hashable view of a call: array shapes and dtypes, scalars as they
    are, and for each state holder its structure, attributes and array
    shapes. Holders are collected in the order the call presents them."""
    if isinstance(value, mx.array):
        return ("array", tuple(value.shape), str(value.dtype))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_signature(v, holders) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple((k, _signature(v, holders)) for k, v in value.items()))
    if value is None or isinstance(value, (int, float, str, bool, mx.Dtype)):
        return value
    if _is_holder(value):
        # Attribute order binds the state arrays to their compiled slots.
        try:
            description = ("state", tuple(vars(value)), state_signature(value))
        except TypeError:
            return ("object", id(value))
        holders.append(value)
        return description
    return ("object", id(value))


def _shared_containers(value, seen=None):
    seen = set() if seen is None else seen
    if isinstance(value, (tuple, list, dict)):
        if id(value) in seen:
            return True
        seen.add(id(value))
        children = value.values() if isinstance(value, dict) else value
        return any(_shared_containers(v, seen) for v in children)
    if _is_holder(value):
        return _shared_containers(vars(value), seen)
    return False


def _state_arrays(value, out=None):
    """Every array a holder reaches, in attribute order, holders inside walked."""
    out = [] if out is None else out
    if isinstance(value, mx.array):
        out.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            _state_arrays(child, out)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _state_arrays(child, out)
    elif _is_holder(value):
        _state_arrays(vars(value), out)
    return out


def _state_template(value, arrays):
    if isinstance(value, mx.array):
        index = len(arrays)
        arrays.append(value)
        return {"$": "ref", "i": index}
    if isinstance(value, (tuple, list)):
        return {"$": type(value).__name__, "items": [_state_template(v, arrays) for v in value]}
    if isinstance(value, dict):
        return {"$": "dict", "items": {k: _state_template(v, arrays) for k, v in value.items()}}
    if _is_holder(value):
        return {"$": "state", "items": {k: _state_template(v, arrays) for k, v in vars(value).items()}}
    return value


def _written(current, template, outputs):
    """The value a holder field takes after a call, built from the template:
    arrays from the compiled outputs, attributes as the traced call left them,
    containers at the traced length, holders the call found updated in place."""
    if isinstance(template, dict) and "$" in template:
        tag = template["$"]
        if tag == "state":
            if not _is_holder(current):
                raise GraphBindingError("a compiled call produced a state holder the scope did not receive")
            fields = vars(current)
            values = {k: _written(fields.get(k), t, outputs) for k, t in template["items"].items()}
            fields.clear()
            fields.update(values)
            return current
        if tag in ("tuple", "list"):
            inside = current if isinstance(current, (tuple, list)) else ()
            values = [_written(inside[i] if i < len(inside) else None, t, outputs)
                      for i, t in enumerate(template["items"])]
            return tuple(values) if tag == "tuple" else values
        if tag == "dict":
            inside = current if isinstance(current, dict) else {}
            return {k: _written(inside.get(k), t, outputs) for k, t in template["items"].items()}
    return _restore(template, outputs)


class GraphWrapper(ReplayWrapper):
    """A thin call router. Subclasses supply portable rule declarations."""

    GRAPH_VARIANTS = ()
    KERNEL_IDS = ()
    GRAPH_BACKEND = True

    def __init__(self, wrapped, specs=None):
        super().__init__(wrapped, specs)
        object.__setattr__(self, "_graph_functions", _GraphCache())
        object.__setattr__(self, "_graph_evidence", [])
        object.__setattr__(self, "_graph_validation", False)
        object.__setattr__(self, "_graph_fallback_reason", "")
        object.__setattr__(self, "_graph_fallbacks", [])  # why a call signature runs the original
        object.__setattr__(self, "_graph_problem", None)

    @contextmanager
    def validate_graph(self):
        """Inspect fresh graphs, bypassing compiled caches for bind checks."""
        old = self._graph_validation
        object.__setattr__(self, "_graph_validation", True)
        self._graph_evidence.clear()
        try:
            yield self._graph_evidence
        finally:
            object.__setattr__(self, "_graph_validation", old)

    def _graph_call(self, args, kwargs):
        if self._graph_fallback_reason:
            return self.wrapped(*args, **kwargs)
        holders = []
        signature = (_signature(args, holders), _signature(kwargs, holders))
        arrays = flatten_arrays((args, kwargs))
        state = _state_arrays([vars(obj) for obj in holders])
        handles = [id(a) for a in arrays + state]
        # Stream handles compare by value but hash by Python identity.
        key = (*signature, tuple(handles.index(h) for h in handles),
               str(mx.default_stream(mx.default_device())))
        entry = self._graph_functions.get(key)
        if entry is None:
            # An unrecorded signature (a cache position never traced) runs
            # the original every time; only compiled entries are kept.
            entry = self._select(args, kwargs, holders, key[2][:len(arrays)])
            if entry is False:
                return self.wrapped(*args, **kwargs)
            self._remember(key, entry)
        elif entry is False:
            return self.wrapped(*args, **kwargs)
        variant, (raw, compiled, templates) = entry
        outputs = execute_graph(raw if self._graph_validation else compiled, arrays, state)
        if self._graph_problem is not None:
            return self._settle(key, args, kwargs)
        for obj, template in zip(holders, templates):
            _written(obj, template, outputs)
        return _restore(variant["outputs"], outputs[:variant["output_count"]])

    def _remember(self, key, entry):
        if len(self._graph_functions) >= GRAPH_CACHE_MAX:
            del self._graph_functions[next(iter(self._graph_functions))]
        self._graph_functions[key] = entry
        return entry

    def _settle(self, key, args, kwargs):
        """A trace found a problem: decide the fallback, then run the original.

        MLX cached that trace with placeholder outputs, so the function is
        dropped either way; a later call with this key builds a fresh one."""
        problem = self._graph_problem
        object.__setattr__(self, "_graph_problem", None)
        self._graph_functions.pop(key, None)
        if self._graph_validation or not isinstance(problem, GraphBindingError):
            raise problem
        self._graph_fallbacks.append(str(problem))
        if isinstance(problem, _PythonStateChanged):
            object.__setattr__(self, "_graph_fallback_reason", str(problem))
            self._graph_functions.clear()
        else:
            # A weight changed shape or dtype, the scope re-traced, and the
            # certified cuts are gone: calls with this signature run the original.
            self._remember(key, False)
        return self.wrapped(*args, **kwargs)

    def _select(self, args, kwargs, holders, aliases):
        """The variant recorded for this call signature, compiled, or False."""
        for variant in self.GRAPH_VARIANTS:
            if not (_matches(variant["args"], args, variant["arg_specs"])
                    and _matches(variant["kwargs"], kwargs, variant["arg_specs"])
                    and list(aliases) == variant["arg_aliases"]):
                continue
            objects = [None] * len(variant["object_signatures"])
            _objects(variant["args"], args, objects)
            _objects(variant["kwargs"], kwargs, objects)
            if ([id(o) for o in objects] != [id(h) for h in holders]
                    or any(json.dumps(state_signature(obj)) != recorded
                           for obj, recorded in zip(objects, variant["object_signatures"]))
                    or _shared_containers([vars(obj) for obj in objects])):
                continue
            if any(not _matches({"$": "ref", "i": 0}, resolve_value(self.wrapped, path), [spec])
                   for path, spec in variant["weight_specs"].items()):
                continue
            return variant, self._make_function(variant, holders)
        return False

    def _make_function(self, variant, objects):
        from . import graph_native

        rules = [(rule, prepare_sequence(rule["sequence"])) for rule in variant["rules"]]
        state_templates = []
        # Freeze the holders' Python structure; every array slot is supplied anew.
        slots = [_Slot(i) for i in range(len(_state_arrays([vars(obj) for obj in objects])))]
        memo = {id(a): slot for a, slot in zip(_state_arrays([vars(obj) for obj in objects]), slots)}
        frozen = copy.deepcopy(objects, memo)

        def transform(arrays, state):
            copies = copy.deepcopy(frozen, {id(slot): state[slot.index] for slot in slots})
            args = _restore(variant["args"], arrays, copies)
            kwargs = _restore(variant["kwargs"], arrays, copies)
            before = _configuration(self.wrapped)
            saved = _python_snapshot(self.wrapped)
            try:
                output = self.wrapped(*args, **kwargs)
            except BaseException:
                _rewind_python(saved)
                raise
            if _configuration(self.wrapped) != before:
                _rewind_python(saved)
                raise _PythonStateChanged("the module changes Python state inside its call")
            roots = flatten_arrays(output)
            # The template also pins nesting and non-array return values.
            specs = [(a.shape, str(a.dtype).removeprefix("mlx.core.")) for a in roots]
            if (len(roots) != variant["output_count"]
                    or not _matches(variant["outputs"], output, specs)):
                raise GraphBindingError("the original module changed its output structure")
            # What the call left in each holder, arrays as further outputs and
            # attributes as values, is written back after every warm call.
            state_templates[:] = [_state_template(obj, roots) for obj in copies]
            evidence = []
            for rule, original in rules:
                inputs = [mx.zeros(shape, getattr(mx, "bool_" if dtype == "bool" else dtype))
                          for shape, dtype in rule["input_specs"]]
                pattern = original(inputs)
                spec = self._specs[rule["kernel_id"]]
                active = not kernels.fallback_fires(spec, inputs)
                anchors = []
                for anchor in rule["anchors"]:
                    anchors.append(None if anchor is None else (
                        arrays[anchor[1]] if anchor[0] == "argument"
                        else resolve_value(self.wrapped, anchor[1])))

                def replace(matched):
                    if any(expected is not None and graph_native.array_id(actual) != graph_native.array_id(expected)
                           for actual, expected in zip(matched, anchors)):
                        return None
                    outputs = kernels.try_call(spec, matched)
                    # A scaffold may stage through extra tmp outputs; the
                    # region's outputs are the prefix the graph receives.
                    return None if outputs is None else outputs[:len(pattern)]

                roots, hits = graph_native.rewrite(roots, pattern, inputs, replace)
                expected = rule["count"] if active else 0
                if hits != expected:
                    raise GraphBindingError(
                        f"kernel {spec.kernel_id}: replaced {hits} graph cuts, expected {expected}; "
                        "a rule replaces every identical occurrence in its scope")
                evidence.append({"kernel_id": spec.kernel_id, "hits": hits,
                                 "expected": expected, "boundary_outputs": len(pattern)})
            self._graph_evidence.append(evidence)
            return roots

        def run(arrays, state):
            try:
                return transform(arrays, state)
            except Exception as problem:
                # MLX puts the module's real arrays back only when the trace
                # returns, so a problem leaves as a flag, never as a raise.
                object.__setattr__(self, "_graph_problem", problem)
                return arrays

        # The module's arrays are compiled-in state: read live on every call
        # by MLX, and a changed shape or dtype re-traces the scope.
        return run, mx.compile(run, inputs=self.wrapped), state_templates
