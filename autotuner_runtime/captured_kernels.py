"""Portable native Metal definitions. Launch arguments remain on trace nodes.

This module deliberately has no optimizer imports: exported wrappers use it too.
"""
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
import json

import mlx.core as mx

_factory = mx.fast.metal_kernel
_recorder = None


def activate(recorder):
    global _recorder
    if _recorder is not None:
        raise RuntimeError("a kernel recorder is already active")
    _recorder = recorder


def deactivate(recorder):
    global _recorder
    if _recorder is recorder:
        _recorder = None


def definition_key(definition):
    return json.dumps(definition, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CapturedKernel:
    def __init__(self, kernel, definition):
        self._kernel = kernel
        self.definition = definition

    def __call__(self, *args, **kwargs):
        rec = _recorder
        if rec is None or not rec.recording:
            return self._kernel(*args, **kwargs)
        with rec.suppressed():
            result = self._kernel(*args, **kwargs)
        count = len(rec.nodes)
        rec.maybe_record("metal_kernel", args, kwargs, result)
        if len(rec.nodes) > count:
            rec.nodes[-1] = replace(rec.nodes[-1], kernel_definition=self.definition)
        return result


def capture(factory, *args, **kwargs):
    # MLX's factory is keyword-only. Let it validate its own API first.
    kernel = factory(*args, **kwargs)
    if isinstance(kernel, CapturedKernel):
        return kernel  # nested capture scopes or an already-installed tracer
    definition = json.loads(definition_key({"version": 1, "kwargs": kwargs}))
    return CapturedKernel(kernel, definition)


@lru_cache(maxsize=256)
def captured(key):
    """Reconstruct once per definition, including compiler and layout options."""
    definition = json.loads(key)
    if definition.get("version") != 1:
        raise ValueError("unsupported captured Metal definition version")
    return CapturedKernel(_factory(**definition["kwargs"]), definition)


@contextmanager
def capture_construction():
    """Capture definitions during a harness-owned build, without recording ops.

    Scoped to construction so timing and ordinary MLX use keep their factory.
    The regular tracer already captures definitions when it is installed.
    """
    if _recorder is not None:
        yield
        return
    original = mx.fast.metal_kernel
    def factory(*args, **kwargs):
        return capture(original, *args, **kwargs)
    mx.fast.metal_kernel = factory
    try:
        yield
    finally:
        mx.fast.metal_kernel = original
