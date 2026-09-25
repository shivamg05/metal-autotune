"""A model with a cache, run as one repeatable step over a fixed context.

A workload's context is how many tokens of conversation are already in place
when the call runs. The cache is built the model's own way (make_cache()),
filled once by running the model over that many tokens, and restored after
every call, so every call is the same step: the capture and pairing laws need
that, and the artifact's load() rebuilds the same step. The model file needs
nothing for this beyond mlx-lm's conventions: make_cache() and a cache=
keyword. Imports nothing from the optimizer.
"""
from __future__ import annotations

import copy

import mlx.core as mx
import mlx.nn as nn


class ContextStep(nn.Module):
    """One call of the model over a cache holding `context` tokens, restored
    afterwards so the next call is the same step."""

    def __init__(self, model, cache, context: int):
        super().__init__()
        self.model = model
        self._cache = cache  # underscored: not a parameter, still reachable for the tracer
        self._context = context

    def __call__(self, *inputs):
        # Copy fields together to preserve aliases between layers. A trim is
        # insufficient for recurrence or a sliding window that overwrote its
        # prefix. Array handles are isolated without eagerly copying buffers.
        snapshots = _copy_cache([vars(c) for c in self._cache])
        try:
            return _with_cache_work(self.model(*inputs, cache=self._cache), self._cache)
        finally:
            # A failure halfway through the layers must not leave the next
            # candidate starting at a different token position.
            for c, saved in zip(self._cache, snapshots):
                vars(c).clear()
                vars(c).update(saved)


    def sequence(self, inputs, steps: int, *, include_state: bool = False):
        """Advance the cache for `steps` calls with the same supplied tokens.

        This is controlled decode, not generated text: the input tokens are
        held fixed so both models do exactly the same work. Each invocation
        starts from an isolated copy of the saved prefix. Copy setup is part
        of the caller's sequence timing. Every output is evaluated, and all
        outputs are returned so correctness can cover the whole trajectory.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("sequence steps must be a positive integer")
        cache = _copy_cache(self._cache)
        outputs = []
        for _ in range(steps):
            out = _with_cache_work(self.model(*inputs, cache=cache), cache)
            from mlx.utils import tree_map
            out = tree_map(lambda a: mx.reshape(a, a.shape) if isinstance(a, mx.array) else a, out)
            mx.eval(out)
            outputs.append(out)
        if include_state:
            return {"outputs": outputs, "state": _cache_observation(cache)}
        return outputs

    def correctness(self, *inputs):
        """Observe one step and its resulting state without changing the prefix."""
        return self.sequence(inputs, 1, include_state=True)


def _cache_state(cache):
    return getattr(cache, "state", vars(cache))


def _with_cache_work(outputs, cache):
    """Evaluating a cached step must also finish its state writes.

    MLX is lazy: a cache-only branch could otherwise disappear when reset
    discards it. Dependencies keep that work in the clock without an extra
    synchronization or changing the model's return structure.
    """
    # The cache's public state exposes live values, excluding unused capacity.
    # Its getter is also a traceable boundary for opaque cache updates.
    pending = [_cache_state(c) for c in cache]
    seen, arrays = set(), []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, mx.array):
            arrays.append(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif hasattr(value, "__dict__"):
            pending.append(vars(value))
    if not arrays:
        return outputs
    from mlx.utils import tree_map
    return tree_map(lambda a: mx.depends(a, arrays) if isinstance(a, mx.array) else a, outputs)


def _cache_observation(cache):
    """Use logical state, excluding unused KV allocation capacity when exposed.

    mlx-lm provides state/meta_state. Small custom caches without that protocol
    are checked conservatively through their instance fields instead.
    """
    def freeze(value):
        if isinstance(value, mx.array):
            result = mx.reshape(value, value.shape)
            mx.eval(result)
            return result
        if isinstance(value, dict):
            return {key: freeze(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(freeze(item) for item in value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if hasattr(value, "__dict__"):
            return freeze(vars(value))
        raise TypeError(f"cache state cannot be checked: {type(value).__name__}")

    observations = []
    for item in cache:
        if hasattr(item, "state"):
            value = {"state": item.state, "meta_state": getattr(item, "meta_state", None),
                     "offset": getattr(item, "offset", None)}
        else:
            value = vars(item)
        observations.append(freeze(value))
    return observations


def correctness_call(model, inputs):
    """Include managed cache state only in correctness checks, never clocks."""
    from .inference import LibraryInference
    if isinstance(model, LibraryInference):
        return model.correctness(*inputs)
    managed = getattr(model, "model", None)
    if isinstance(managed, LibraryInference):
        return managed.correctness(*inputs)
    if isinstance(model, ContextStep):
        return model.correctness(*inputs)
    if isinstance(model, ContextSequence):
        return model.step.sequence(inputs, model.steps, include_state=True)
    return model(*inputs)


def sequence_observation(model, inputs, steps):
    """Save every step's result for correctness, including managed final state."""
    from .inference import LibraryInference
    if isinstance(model, LibraryInference):
        return model.correctness(*inputs)
    managed = getattr(model, "model", None)
    if isinstance(managed, LibraryInference):
        return managed.correctness(*inputs)
    if isinstance(model, ContextStep):
        return model.sequence(inputs, steps, include_state=True)
    if isinstance(managed, ContextStep):
        return managed.sequence(inputs, steps, include_state=True)
    if isinstance(model, ContextSequence):
        return model.step.sequence(inputs, steps, include_state=True)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("sequence steps must be a positive integer")

    def snapshot(tree):
        if isinstance(tree, mx.array):
            result = mx.array(tree)
            mx.eval(result)
            return result
        if isinstance(tree, dict):
            return {key: snapshot(value) for key, value in tree.items()}
        if isinstance(tree, (list, tuple)):
            return type(tree)(snapshot(value) for value in tree)
        return copy.deepcopy(tree)

    return [snapshot(model(*inputs)) for _ in range(steps)]


def _copy_cache(cache):
    """Copy cache metadata and array handles without changing the live graph.

    MLX indexed writes update an array object's graph. New handles isolate
    these writes while sharing the immutable backing data until it changes.
    Seeding deepcopy's memo also preserves aliases between cache fields.
    Same-shape reshape creates a distinct handle without copying data and
    keeps these copies visible to the precision auditor.
    """
    memo, seen = {}, set()

    def arrays(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, mx.array):
            memo[id(value)] = mx.reshape(value, value.shape)
        elif isinstance(value, dict):
            for item in value.values():
                arrays(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                arrays(item)
        elif hasattr(value, "__dict__"):
            arrays(vars(value))

    arrays(cache)
    return copy.deepcopy(cache, memo)


class ContextSequence(nn.Module):
    """An advancing decode workload with a repeatable starting prefix."""

    def __init__(self, step, steps, *, include_state=False):
        super().__init__()
        self.step = step
        self.steps = steps
        self._include_state = include_state

    def __call__(self, *inputs):
        return self.step.sequence(inputs, self.steps, include_state=self._include_state)


def _make_cache(model):
    """The cache the model's own way: mlx-lm's rule, make_cache() when the
    model defines it, else its default cache, one per layer."""
    if callable(getattr(model, "make_cache", None)):
        return model.make_cache()
    try:
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(model)
    except (ImportError, AttributeError, TypeError):
        raise TypeError("a workload with a context needs a model that defines make_cache() or "
                        "an mlx_lm model; for any other kind of state, wrap the step in build()")


def context_step(model, context: int, tokens, warm) -> ContextStep:
    """Build the model's cache from a prefix, then exercise a repeatable call.

    Reset restores tensors, positions and allocation state together. Empty
    cache allocation and later growth therefore remain part of the workload.
    """
    cache = _make_cache(model)
    if any(not hasattr(c, "__dict__") for c in cache):
        raise TypeError("this model's cache cannot be restored; provide a cache with Python "
                        "state fields")
    if context:
        mx.eval(_with_cache_work(model(tokens, cache=cache), cache))
    step = ContextStep(model, cache, context)
    mx.eval(step(*warm))
    return step
