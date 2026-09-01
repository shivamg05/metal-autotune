"""The op name table: op name <-> callable, built by importing mx.

Both the patch installer and replay consume this one table (plan 5.4), so
replayed-in-process and generated-as-source share one semantics, and replay
works in a patch-free subprocess. Names look like "mx.add", "mx.fast.rms_norm",
"array.__add__", "array.reshape".
"""

from __future__ import annotations

import mlx.core as mx

# Callables that are not ops: control, transforms, IO, device plumbing. The
# tracer must never wrap these; some are special-cased (eval, compile).
SKIP_FUNCTIONS = frozenset({
    "eval", "async_eval", "synchronize", "compile", "disable_compile",
    "enable_compile", "grad", "value_and_grad", "vmap", "custom_function",
    "custom_vjp", "checkpoint", "stop_gradient_fn", "export_function",
    "import_function", "exporter", "eval_shapeless",
    "save", "savez", "savez_compressed", "load", "save_safetensors",
    "save_gguf", "set_default_device", "default_device", "set_default_stream",
    "default_stream", "new_stream", "stream", "cpu", "gpu", "device_info",
    "get_active_memory", "get_peak_memory", "reset_peak_memory",
    "get_cache_memory", "set_memory_limit", "set_cache_limit",
    "set_wired_limit", "clear_cache", "issubdtype", "finfo", "iinfo",
    "enable_tf32", "cuda", "distributed", "metal", "linalg", "fast", "random",
    "fft", "array", "Dtype", "Device", "DtypeCategory", "Stream", "finfo",
})

# Dunders whose call means the model observed a value: recording one while
# armed marks in-pass evaluation. __setitem__ and the in-place ops mutate their
# first argument; the recorder renames the mutated object (SSA).
EVAL_METHODS = frozenset({"item", "tolist", "__bool__", "__int__", "__float__", "__index__"})
MUTATING_METHODS = frozenset({
    "__setitem__", "__iadd__", "__isub__", "__imul__", "__itruediv__",
    "__ifloordiv__", "__imod__", "__ipow__", "__imatmul__", "__iand__",
    "__ior__", "__ixor__", "__ilshift__", "__irshift__",
})

# The array surface (spike_01 inventory): every present dunder plus the public
# array-returning methods. Non-existent reflected dunders raise unpatched too,
# so they are not holes.
ARRAY_DUNDERS = (
    "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__",
    "__pow__", "__matmul__", "__and__", "__or__", "__xor__", "__lshift__", "__rshift__",
    "__radd__", "__rsub__", "__rmul__", "__rtruediv__", "__rfloordiv__", "__rmod__", "__rpow__",
    "__iadd__", "__isub__", "__imul__", "__itruediv__", "__ifloordiv__", "__imod__",
    "__ipow__", "__imatmul__", "__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__",
    "__neg__", "__abs__", "__invert__",
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__getitem__", "__setitem__",
    "__bool__", "__int__", "__float__",
)

_MODULES = (("mx", mx), ("mx.fast", mx.fast), ("mx.linalg", mx.linalg),
            ("mx.random", mx.random), ("mx.fft", mx.fft))


def module_function_names() -> list[str]:
    """Every patchable public callable on the op modules."""
    names = []
    for prefix, mod in _MODULES:
        for attr in dir(mod):
            if attr.startswith("_") or attr in SKIP_FUNCTIONS:
                continue
            value = getattr(mod, attr)
            if callable(value) and not isinstance(value, type):
                names.append(f"{prefix}.{attr}")
    return names


# Array-returning data descriptors: read through a property, not a call, so
# they need their own patch kind (the completeness check caught k.T unrecorded).
ARRAY_PROPERTIES = ("T", "real", "imag")


def array_property_names() -> list[str]:
    return [f"array.{p}" for p in ARRAY_PROPERTIES if p in vars(mx.array)]


def array_method_names() -> list[str]:
    """Present dunders plus public methods on mx.array."""
    names = [d for d in ARRAY_DUNDERS if d in vars(mx.array)]
    for attr, value in vars(mx.array).items():
        if attr.startswith("_") or not callable(value):
            continue
        names.append(attr)
    return [f"array.{n}" for n in names]


# Originals saved by the patch installer, so resolve() stays patch-invisible
# in a patched process and works identically in a patch-free subprocess.
_ORIGINALS: dict[str, object] = {}


def register_original(op_name: str, fn: object) -> None:
    _ORIGINALS[op_name] = fn


def clear_originals() -> None:
    _ORIGINALS.clear()


def resolve(op_name: str):
    """Op name -> callable. Prefers the pre-patch original when a tracer is
    installed; otherwise resolves live from mx (the patch-free subprocess case)."""
    if op_name in _ORIGINALS:
        return _ORIGINALS[op_name]
    if op_name.startswith("array."):
        attr = getattr(mx.array, op_name.removeprefix("array."))
        if not callable(attr):
            return lambda a, _d=attr: _d.__get__(a, mx.array)  # data descriptor
        return attr
    parts = op_name.split(".")
    if parts[0] != "mx":
        raise KeyError(f"unknown op name {op_name!r}")
    obj = mx
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj
