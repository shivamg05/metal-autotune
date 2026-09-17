"""The patch surface: mx module functions, array dunders and
methods, mx.compile, mx.eval, and per-subclass Module.__call__.

Install BEFORE the model file is imported; a model that imported an op earlier
would keep the unwrapped original. Uninstall restores every attribute to the
identical original object.

One known hole, by construction: mx.array is a nanobind type, and replacing
the name would break isinstance checks everywhere, so a model that constructs
arrays via mx.array(...) inside its forward yields a completeness abort naming
the call; the fix is hoisting the constant or using mx.zeros/ones/full/arange,
which are patched functions.
"""

from __future__ import annotations

import functools
import inspect
import sys
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from .optable import (
    EVAL_METHODS,
    MUTATING_METHODS,
    array_method_names,
    array_property_names,
    clear_originals,
    module_function_names,
    register_original,
    resolve,
)
from .recorder import Recorder, compiled_op
from autotuner_runtime import captured_kernels
from .walk import state_holders


def walk_module_paths(model: object) -> dict[int, str]:
    """id(module instance) -> dot path, for every nn.Module in the tree."""
    paths: dict[int, str] = {}
    if not isinstance(model, nn.Module):
        return paths
    paths[id(model)] = ""

    def visit(m: nn.Module, prefix: str) -> None:
        for key, value in m.items():
            child_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, nn.Module):
                paths.setdefault(id(value), child_path)
                visit(value, child_path)
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if isinstance(v, nn.Module):
                        paths.setdefault(id(v), f"{child_path}.{i}")
                        visit(v, f"{child_path}.{i}")
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, nn.Module):
                        paths.setdefault(id(v), f"{child_path}.{k}")
                        visit(v, f"{child_path}.{k}")

    visit(model, "")
    return paths


class _CompiledProxy:
    """Stands in for a compiled callable. While recording it runs the compiled
    function with recording suppressed and records one node for the whole
    call, named by the function's import path when it has one, so a scope
    containing it can still be replayed by calling that path. Disarmed, the
    only cost is one flag check per call."""

    def __init__(self, compiled: Callable, recorder: Recorder, path: str | None = None):
        self._compiled = compiled
        self._recorder = recorder
        self._path = path

    def __call__(self, *args, **kwargs):
        rec = self._recorder
        if not rec.recording:
            return self._compiled(*args, **kwargs)
        with rec.suppressed():
            result = self._compiled(*args, **kwargs)
        rec.maybe_record(compiled_op(self._path), args, kwargs, result)
        return result


def _import_path(fun) -> str | None:
    """module.name for a function reachable by import; None for lambdas,
    closures, and methods, which nothing outside their scope can find."""
    module, name = getattr(fun, "__module__", None), getattr(fun, "__qualname__", "")
    if not module or not name or "<" in name or "." in name:
        return None
    return f"{module}.{name}"


class Patcher:
    def __init__(self, recorder: Recorder):
        self.recorder = recorder
        self._fn_originals: dict[str, Callable] = {}
        self._method_originals: dict[str, Callable] = {}
        self._property_originals: dict[str, object] = {}
        self._special_originals: dict[str, Callable] = {}
        self._precompiled_originals: list[tuple[object, str, object]] = []
        self._class_originals: list[tuple[type, bool, Callable]] = []
        self._holder_originals: list[tuple[type, str, Callable]] = []
        self.installed = False

    # -- global surface, before model import ---------------------------------

    def install(self, model_module_name: str | None = None) -> None:
        if self.installed:
            raise RuntimeError("patch surface already installed")
        if model_module_name and model_module_name in sys.modules:
            raise RuntimeError(
                f"{model_module_name} is already imported; the tracer must install "
                f"before the model file is imported"
            )
        captured_kernels.activate(self.recorder)
        self.installed = True
        try:
            for name in module_function_names():
                self._patch_function(name)
            for name in array_method_names():
                self._patch_array_method(name)
            for name in array_property_names():
                self._patch_array_property(name)
            self._patch_specials()
            self._patch_precompiled()
        except BaseException:
            self.uninstall()
            raise

    def _patch_precompiled(self) -> None:
        """Record imported compiled helpers, including ones loaded by a prior
        model in this process. Retarget old proxies to this recorder too."""
        compiled_type = type(self._special_originals["compile"](lambda x: x))
        recorder = self.recorder
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            for attr, value in list(vars(mod).items()):
                if type(value) in (compiled_type, _CompiledProxy):
                    self._precompiled_originals.append((mod, attr, value))
                    compiled = value._compiled if isinstance(value, _CompiledProxy) else value
                    path = value._path if isinstance(value, _CompiledProxy) else f"{mod_name}.{attr}"
                    setattr(mod, attr, _CompiledProxy(compiled, recorder, path))

    def _patch_array_property(self, op_name: str) -> None:
        short = op_name.removeprefix("array.")
        orig = vars(mx.array)[short]
        recorder = self.recorder

        def getter(self_arr):
            result = orig.__get__(self_arr, mx.array)
            recorder.maybe_record(op_name, (self_arr,), {}, result)
            return result

        self._property_originals[op_name] = orig
        register_original(op_name, lambda a, _d=orig: _d.__get__(a, mx.array))
        setattr(mx.array, short, property(getter))

    def _patch_function(self, op_name: str) -> None:
        parts = op_name.split(".")
        mod = mx
        for part in parts[1:-1]:
            mod = getattr(mod, part)
        orig = getattr(mod, parts[-1])
        recorder = self.recorder

        def wrapper(*args, **kwargs):
            result = orig(*args, **kwargs)
            recorder.maybe_record(op_name, args, kwargs, result)
            return result

        wrapper.__name__ = getattr(orig, "__name__", parts[-1])
        self._fn_originals[op_name] = orig
        register_original(op_name, orig)
        setattr(mod, parts[-1], wrapper)

    def _patch_array_method(self, op_name: str) -> None:
        short = op_name.removeprefix("array.")
        orig = getattr(mx.array, short)
        recorder = self.recorder

        if short in EVAL_METHODS:
            def wrapper(self_arr, *args, **kwargs):
                recorder.note_evaluation(self_arr)
                return orig(self_arr, *args, **kwargs)
        else:
            mutates = short in MUTATING_METHODS

            def wrapper(self_arr, *args, **kwargs):
                result = orig(self_arr, *args, **kwargs)
                if result is NotImplemented:
                    return result
                recorder.maybe_record(
                    op_name, (self_arr, *args), kwargs, result, mutates_first=mutates
                )
                return result

        wrapper.__name__ = short
        self._method_originals[op_name] = orig
        register_original(op_name, orig)
        setattr(mx.array, short, wrapper)

    def _patch_specials(self) -> None:
        import autotuner_runtime.kernels as rt_kernels
        import autotuner_runtime.state as rt_state
        import autotuner_runtime.graph as rt_graph

        recorder = self.recorder
        orig_compile = mx.compile
        orig_eval = mx.eval
        orig_async_eval = mx.async_eval
        orig_kernel_call = rt_kernels.call
        orig_metal_kernel = mx.fast.metal_kernel
        orig_cache_state = rt_state._cache_state
        orig_graph_call = rt_graph.execute_graph
        self._special_originals = {
            "compile": orig_compile, "eval": orig_eval, "async_eval": orig_async_eval,
            "kernel_call": orig_kernel_call, "metal_kernel": orig_metal_kernel,
            "cache_state": orig_cache_state,
            "graph_call": orig_graph_call,
        }

        def graph_call_wrapper(function, arrays, state):
            if not recorder.recording:
                return orig_graph_call(function, arrays, state)
            with recorder.suppressed():
                result = orig_graph_call(function, arrays, state)
            recorder.maybe_record(compiled_op(None), (arrays, state), {}, result)
            return result

        rt_graph.execute_graph = graph_call_wrapper

        def cache_state_wrapper(cache):
            if not recorder.recording or id(cache) not in recorder.state_holders:
                return orig_cache_state(cache)
            # Reading state is an opaque boundary just like updating it.
            # Internal allocation buffers need not have appeared as outputs
            # of update_and_fetch for this live read to be replayable.
            recorder.state_enter(cache)
            try:
                with recorder.suppressed():
                    name = "state" if hasattr(cache, "state") else "__dict__"
                    result = getattr(cache, name)
            except BaseException:
                recorder.state_abort()
                raise
            recorder.state_exit(cache, "__getattribute__", (name,), {}, result)
            return result

        rt_state._cache_state = cache_state_wrapper

        def kernel_call_wrapper(spec, inputs, init_value=None, launch=None):
            # the harness's own kernel is one custom_kernel node; the stand-in
            # its metal_kernel object became must not record a second one
            with recorder.suppressed():
                result = orig_kernel_call(spec, inputs, init_value=init_value, launch=launch)
            recorder.maybe_record(
                "custom_kernel", tuple(inputs), {"kernel_id": spec.kernel_id}, result
            )
            return result

        rt_kernels.call = kernel_call_wrapper

        def compile_wrapper(fun, *args, **kwargs):
            return _CompiledProxy(orig_compile(fun, *args, **kwargs), recorder, _import_path(fun))

        def metal_kernel_wrapper(*args, **kwargs):
            return captured_kernels.capture(orig_metal_kernel, *args, **kwargs)

        def eval_wrapper(*args):
            recorder.note_evaluation(*args)
            return orig_eval(*args)

        def async_eval_wrapper(*args):
            recorder.note_evaluation(*args)
            return orig_async_eval(*args)

        mx.compile = compile_wrapper
        mx.eval = eval_wrapper
        mx.async_eval = async_eval_wrapper
        mx.fast.metal_kernel = metal_kernel_wrapper

    # -- model surface, after build() ----------------------------------------

    def wrap_model(self, model: object) -> dict[int, str]:
        """Patch type(m).__call__ once per distinct Module subclass in the
        tree, dispatching on self identity. Returns the instance path map for
        Recorder.arm. The root instance is transparent: step() addresses it."""
        paths = walk_module_paths(model)
        self._patch_state_holders(model)
        recorder = self.recorder
        patched: set[type] = {cls for cls, _, _ in self._class_originals}
        for m in _all_modules(model):
            cls = type(m)
            if cls in patched:
                continue
            owner = _mro_call_owner(cls)
            if owner is None:
                continue
            orig = owner.__dict__["__call__"]
            had_own = "__call__" in cls.__dict__

            def make_wrapper(orig_call):
                def wrapper(self_mod, *args, **kwargs):
                    if not recorder.recording or recorder.is_root(self_mod):
                        return orig_call(self_mod, *args, **kwargs)
                    recorder.module_enter(self_mod, args, kwargs)
                    try:
                        result = orig_call(self_mod, *args, **kwargs)
                    except BaseException:
                        recorder.module_abort()
                        raise
                    recorder.module_exit(result)
                    return result
                return wrapper

            self._class_originals.append((cls, had_own, cls.__dict__.get("__call__")))
            cls.__call__ = make_wrapper(orig)
            patched.add(cls)
        return paths

    def _patch_state_holders(self, model: object) -> None:
        """Every plain object holding arrays (a KV cache, a helper with a
        table) gets its methods wrapped, once per class, so the recorder can
        tell whether a call changed the object's state and collapse it into
        one state call if it did."""
        holders = state_holders(model)
        recorder = self.recorder
        recorder.state_holders = {id(obj): path for path, obj in holders}
        wrapped = {(klass, name) for klass, name, _ in self._holder_originals}
        for _path, obj in holders:
            for klass in type(obj).__mro__:
                if klass is object:
                    continue
                for name, fn in list(vars(klass).items()):
                    if (klass, name) in wrapped:
                        continue
                    if name == "state" and isinstance(fn, property):
                        # Reading state is a boundary like writing it: a
                        # library's generation loop evaluates it directly.
                        setattr(klass, name, property(_state_call(fn.fget, name, recorder), fn.fset, fn.fdel))
                    elif inspect.isfunction(fn) and not (
                            name.startswith("_") and name not in {"__getitem__", "__setitem__"}):
                        setattr(klass, name, _state_call(fn, name, recorder))
                    else:
                        continue
                    self._holder_originals.append((klass, name, fn))
                    wrapped.add((klass, name))

    def unwrap_model(self) -> None:
        for cls, had_own, orig in reversed(self._class_originals):
            if had_own:
                cls.__call__ = orig
            else:
                del cls.__call__
        self._class_originals.clear()
        for klass, name, fn in reversed(self._holder_originals):
            setattr(klass, name, fn)
        self._holder_originals.clear()

    # -- teardown -------------------------------------------------------------

    def uninstall(self) -> None:
        if not self.installed:
            return
        captured_kernels.deactivate(self.recorder)
        self.unwrap_model()
        for op_name, orig in self._fn_originals.items():
            parts = op_name.split(".")
            mod = mx
            for part in parts[1:-1]:
                mod = getattr(mod, part)
            setattr(mod, parts[-1], orig)
        for op_name, orig in self._method_originals.items():
            setattr(mx.array, op_name.removeprefix("array."), orig)
        for op_name, orig in self._property_originals.items():
            setattr(mx.array, op_name.removeprefix("array."), orig)
        self._property_originals.clear()
        import autotuner_runtime.kernels as rt_kernels
        import autotuner_runtime.state as rt_state
        import autotuner_runtime.graph as rt_graph

        for mod, attr, value in self._precompiled_originals:
            setattr(mod, attr, value)
        self._precompiled_originals.clear()
        mx.compile = self._special_originals["compile"]
        mx.eval = self._special_originals["eval"]
        mx.async_eval = self._special_originals["async_eval"]
        mx.fast.metal_kernel = self._special_originals["metal_kernel"]
        rt_kernels.call = self._special_originals["kernel_call"]
        rt_state._cache_state = self._special_originals["cache_state"]
        rt_graph.execute_graph = self._special_originals["graph_call"]
        self._fn_originals.clear()
        self._method_originals.clear()
        self._special_originals.clear()
        clear_originals()
        self.installed = False

    def verify_restored(self) -> list[str]:
        """Identity check on every patched attribute; empty list means clean."""
        bad = []
        from autotuner_runtime.graph import execute_graph
        if getattr(execute_graph, "__module__", "") == __name__:
            bad.append("graph.execute_graph")
        for name in module_function_names():
            if getattr(resolve(name), "__module__", "") == __name__:
                bad.append(name)
        for name in array_method_names():
            if getattr(resolve(name), "__module__", "") == __name__:
                bad.append(name)
        for name, value in (("mx.compile", mx.compile), ("mx.eval", mx.eval),
                            ("mx.async_eval", mx.async_eval), ("mx.fast.metal_kernel", mx.fast.metal_kernel)):
            if getattr(value, "__module__", "") == __name__:
                bad.append(name)
        return bad


def _state_call(orig: Callable, name: str, recorder: Recorder) -> Callable:
    @functools.wraps(orig)
    def wrapper(self_obj, *args, **kwargs):
        if not recorder.recording or id(self_obj) not in recorder.state_holders:
            return orig(self_obj, *args, **kwargs)
        recorder.state_enter(self_obj)
        try:
            result = orig(self_obj, *args, **kwargs)
        except BaseException:
            recorder.state_abort()
            raise
        recorder.state_exit(self_obj, name, args, kwargs, result)
        return result
    return wrapper


def _all_modules(model: object):
    if not isinstance(model, nn.Module):
        return []
    out = [model]
    seen = {id(model)}

    def visit(m: nn.Module) -> None:
        for value in m.values():
            children = []
            if isinstance(value, nn.Module):
                children = [value]
            elif isinstance(value, (list, tuple)):
                children = [v for v in value if isinstance(v, nn.Module)]
            elif isinstance(value, dict):
                children = [v for v in value.values() if isinstance(v, nn.Module)]
            for c in children:
                if id(c) not in seen:
                    seen.add(id(c))
                    out.append(c)
                    visit(c)

    visit(model)
    return out


def _mro_call_owner(cls: type) -> type | None:
    """The class in the MRO that actually defines __call__ (never nn.Module,
    which has none)."""
    for klass in cls.__mro__:
        if "__call__" in klass.__dict__:
            return klass
    return None
