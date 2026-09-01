"""The patch surface (plan 5.1): mx module functions, array dunders and
methods, mx.compile, mx.eval, and per-subclass Module.__call__.

Install BEFORE the model file is imported; a model that imported an op earlier
would keep the unwrapped original. Uninstall restores every attribute to the
identical original object (spike_01 pins that this works).

One known hole, by construction: mx.array is a nanobind type, and replacing
the name would break isinstance checks everywhere, so a model that constructs
arrays via mx.array(...) inside its forward yields a completeness abort naming
the call; the fix is hoisting the constant or using mx.zeros/ones/full/arange,
which are patched functions.
"""

from __future__ import annotations

import sys
from pathlib import Path
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
from .recorder import OPAQUE_OP, Recorder


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
    """Stands in for a model-owned compiled callable. A compile the tracer
    witnessed (it holds the plain function) records as that function's own
    ops while armed and runs compiled when disarmed, so scopes containing it
    stay replayable; identity certification stays the bitwise arbiter before
    any swap. A compile predating the tracer records as one opaque node
    (spike_09: a compiled section's trace reruns its body under suppression).
    Disarmed, the only cost is one flag check per compiled call."""

    def __init__(self, compiled: Callable, recorder: Recorder, plain: Callable | None = None):
        self._compiled = compiled
        self._recorder = recorder
        self._plain = plain

    def __call__(self, *args, **kwargs):
        rec = self._recorder
        if not rec.recording:
            return self._compiled(*args, **kwargs)
        if self._plain is not None:
            return self._plain(*args, **kwargs)
        with rec.suppressed():
            result = self._compiled(*args, **kwargs)
        rec.maybe_record(OPAQUE_OP, args, kwargs, result)
        return result


class Patcher:
    def __init__(self, recorder: Recorder):
        self.recorder = recorder
        self._fn_originals: dict[str, Callable] = {}
        self._method_originals: dict[str, Callable] = {}
        self._property_originals: dict[str, object] = {}
        self._special_originals: dict[str, Callable] = {}
        self._precompiled_originals: list[tuple[object, str, object]] = []
        self._class_originals: list[tuple[type, bool, Callable]] = []
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
        for name in module_function_names():
            self._patch_function(name)
        for name in array_method_names():
            self._patch_array_method(name)
        for name in array_property_names():
            self._patch_array_property(name)
        self._patch_specials()
        self._patch_precompiled()
        self.installed = True

    def _patch_precompiled(self) -> None:
        """mlx.nn ships every activation compiled at import time (nn.silu and
        friends), before any tracer could wrap mx.compile. Calling one while
        armed leaks its compile-trace placeholder arrays into the record, and
        an opaque proxy would strand every scope containing an activation. So
        the library's own activations source is re-executed with the compile
        decorator neutralized, yielding the identical math as plain recordable
        ops; those substitute in. Anything compiled we cannot recover this way
        falls back to the opaque proxy (correct, merely stranding)."""
        compiled_type = type(self._special_originals["compile"](lambda x: x))
        recorder = self.recorder
        plain = self._plain_activations()
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not mod_name.startswith("mlx.nn"):
                continue
            for attr, value in list(vars(mod).items()):
                if isinstance(value, compiled_type):
                    self._precompiled_originals.append((mod, attr, value))
                    replacement = plain.get(attr) or _CompiledProxy(value, recorder)
                    setattr(mod, attr, replacement)

    def _plain_activations(self) -> dict:
        """Re-exec mlx/nn/layers/activations.py with mx.compile as identity."""
        import importlib.util

        source_file = Path(nn.layers.activations.__file__)
        try:
            spec = importlib.util.spec_from_file_location("autotuner_plain_activations", source_file)
            module = importlib.util.module_from_spec(spec)
            saved = mx.compile
            mx.compile = lambda fun, *a, **k: fun
            try:
                spec.loader.exec_module(module)
            finally:
                mx.compile = saved
            return {
                k: v for k, v in vars(module).items()
                if callable(v) and not isinstance(v, type) and not k.startswith("_")
            }
        except Exception:
            return {}

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
                recorder.note_evaluation()
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

        recorder = self.recorder
        orig_compile = mx.compile
        orig_eval = mx.eval
        orig_async_eval = mx.async_eval
        orig_kernel_call = rt_kernels.call
        self._special_originals = {
            "compile": orig_compile, "eval": orig_eval, "async_eval": orig_async_eval,
            "kernel_call": orig_kernel_call,
        }

        def kernel_call_wrapper(spec, inputs, init_value=None):
            result = orig_kernel_call(spec, inputs, init_value=init_value)
            recorder.maybe_record(
                "custom_kernel", tuple(inputs), {"kernel_id": spec.kernel_id}, result
            )
            return result

        rt_kernels.call = kernel_call_wrapper

        def compile_wrapper(fun, *args, **kwargs):
            return _CompiledProxy(orig_compile(fun, *args, **kwargs), recorder, plain=fun)

        def eval_wrapper(*args):
            recorder.note_evaluation()
            return orig_eval(*args)

        def async_eval_wrapper(*args):
            recorder.note_evaluation()
            return orig_async_eval(*args)

        mx.compile = compile_wrapper
        mx.eval = eval_wrapper
        mx.async_eval = async_eval_wrapper

    # -- model surface, after build() ----------------------------------------

    def wrap_model(self, model: object) -> dict[int, str]:
        """Patch type(m).__call__ once per distinct Module subclass in the
        tree, dispatching on self identity. Returns the instance path map for
        Recorder.arm. The root instance is transparent: step() addresses it."""
        paths = walk_module_paths(model)
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

    def unwrap_model(self) -> None:
        for cls, had_own, orig in reversed(self._class_originals):
            if had_own:
                cls.__call__ = orig
            else:
                del cls.__call__
        self._class_originals.clear()

    # -- teardown -------------------------------------------------------------

    def uninstall(self) -> None:
        if not self.installed:
            return
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

        for mod, attr, value in self._precompiled_originals:
            setattr(mod, attr, value)
        self._precompiled_originals.clear()
        mx.compile = self._special_originals["compile"]
        mx.eval = self._special_originals["eval"]
        mx.async_eval = self._special_originals["async_eval"]
        rt_kernels.call = self._special_originals["kernel_call"]
        self._fn_originals.clear()
        self._method_originals.clear()
        self._special_originals.clear()
        clear_originals()
        self.installed = False

    def verify_restored(self) -> list[str]:
        """Identity check on every patched attribute; empty list means clean."""
        bad = []
        for name in module_function_names():
            if getattr(resolve(name), "__module__", "") == __name__:
                bad.append(name)
        for name in array_method_names():
            if getattr(resolve(name), "__module__", "") == __name__:
                bad.append(name)
        return bad


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
    which has none; spike_02)."""
    for klass in cls.__mro__:
        if "__call__" in klass.__dict__:
            return klass
    return None
