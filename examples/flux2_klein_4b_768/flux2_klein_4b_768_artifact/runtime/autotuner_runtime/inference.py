"""Repeatable native library inference, shared by search and exported bundles.

One call completes a fixed number of generated tokens. It excludes model loading,
tokenization and any pre-existing context, and includes the library's normal
sampling, cache updates and asynchronous work. No optimizer imports belong here.
"""

from __future__ import annotations

import inspect
import sys
from importlib.util import find_spec

import mlx.core as mx
import mlx.nn as nn

from .state import _cache_observation, _copy_cache, _with_cache_work


def resolve_library_inference(model, requested, workloads, *, dims=None, sweep=None) -> bool:
    """Resolve once before tracing; an explicit unsupported request is an error.

    Recognition uses the installed library's model classes and call protocol,
    never architecture names or workload labels. A user-defined subclass of a
    library model keeps its supported provenance. Arbitrary callables retain
    forward measurement because a cache argument alone does not mean generation.
    """
    if requested is not None and not isinstance(requested, bool):
        raise ValueError("use_library_inference must be true or false")
    if requested is False:
        return False
    reason = None
    # MLX-LM exports its complete model as Model (TextModel for the standalone
    # language part of multimodal definitions). An internal transformer body can
    # also have layers/cache while returning hidden states instead of logits.
    supported_class = any(
        cls.__module__.startswith("mlx_lm.models.")
        and any(getattr(sys.modules.get(cls.__module__), name, None) is cls
                for name in ("Model", "TextModel"))
        for cls in type(model).__mro__)
    if not supported_class:
        reason = "the returned model is not a complete MLX-LM language model"
    elif find_spec("mlx_lm") is None:
        reason = "mlx-lm is not installed"
    elif not hasattr(model, "layers"):
        reason = "the returned model does not expose MLX-LM's layer/cache protocol"
    else:
        try:
            params = inspect.signature(model.__call__).parameters.values()
            if not any(p.name == "cache" or p.kind == p.VAR_KEYWORD for p in params):
                reason = "the returned model cannot accept MLX-LM's cache argument"
        except (TypeError, ValueError):
            reason = "the model's cache call contract cannot be inspected"
    if reason is None:
        for workload in workloads:
            specs = workload.inputs
            if len(specs) != 1 or specs[0].dtype not in ("int32", "uint32", "int64", "uint64"):
                reason = "MLX-LM inference needs one integer token input"
                break
            shape = tuple((dims or {}).get(d, d) for d in specs[0].shape)
            if not (len(shape) == 1 or len(shape) == 2 and shape[0] == 1):
                reason = "MLX-LM generate_step supports one prompt: use shape [T] or [1, T]"
                break
            batch_dim = specs[0].shape[0] if len(shape) == 2 else None
            if isinstance(batch_dim, str) and any(size != 1 for size in (sweep or {}).get(batch_dim, ())):
                reason = "MLX-LM generate_step does not support a correctness sweep over multiple prompts"
                break
    if reason is not None:
        if requested:
            raise ValueError(f"use_library_inference: true is unsupported: {reason}; "
                             "set it to false to measure the model's forward call")
        return False
    return True


def _prompt(inputs):
    if len(inputs) != 1:
        raise ValueError("library inference needs one token input")
    tokens = inputs[0]
    if not isinstance(tokens, mx.array) or tokens.dtype not in (mx.int32, mx.uint32, mx.int64, mx.uint64):
        raise ValueError("library inference needs integer token ids")
    if tokens.ndim == 2 and tokens.shape[0] == 1:
        tokens = tokens[0]
    if tokens.ndim != 1 or tokens.size == 0:
        raise ValueError("library inference needs a nonempty prompt of shape [T] or [1, T]")
    return tokens


class LibraryInference(nn.Module):
    """Run unmodified MLX-LM generation with independent state for each trial.

    The underlying model remains an ordinary child module, so installed kernels,
    weight updates and tracing see the exact model that inference calls. This
    Python generator must not itself be passed to mx.compile.
    """

    def __init__(self, model, *, steps: int, prefix_tokens=None):
        super().__init__()
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("inference steps must be a positive integer")
        self.model = model
        self.steps = steps
        self._prefix_tokens = prefix_tokens
        self.refresh_prefix()

    def refresh_prefix(self):
        """Rebuild prepared context after an explicit change to model weights."""
        from mlx_lm.generate import generate_step, generation_stream
        from mlx_lm.models.cache import make_prompt_cache

        # One cache object per layer for the runtime's life, reset in place
        # before every trial: the tracer registers these holders once, as it
        # does a ContextStep's, and every trial still starts from the same state.
        self._cache = make_prompt_cache(self.model)
        prefix_tokens = self._prefix_tokens
        if prefix_tokens is not None and prefix_tokens.size:
            # A zero-output library call consumes exactly the supplied context.
            # Preparation happens once, before any timed trial.
            list(generate_step(_prompt([prefix_tokens]), self.model, max_tokens=0, prompt_cache=self._cache))
            mx.eval([item.state for item in self._cache])
            mx.synchronize(generation_stream)
        self._start = _copy_cache([vars(c) for c in self._cache])

    def _reset(self):
        """Fresh array handles from the prepared context, so a trial's in-place
        cache writes never reach the saved state."""
        for c, saved in zip(self._cache, _copy_cache(self._start)):
            vars(c).clear()
            vars(c).update(saved)

    def _run(self, inputs, *, include_state=False):
        from mlx_lm.generate import generate_step, generation_stream

        prompt = _prompt(inputs)
        self._reset()
        cache = self._cache
        # Exhaust the actual generator. Its lookahead and scheduling remain the
        # library's responsibility, including work queued before the final yield.
        try:
            generated = list(generate_step(prompt, self.model, max_tokens=self.steps, prompt_cache=cache))
            if len(generated) != self.steps:
                raise RuntimeError(f"library inference returned {len(generated)} tokens; expected {self.steps}")
            # The cache's state is read through the boundary the tracer
            # records as a state call, and finishes with the outputs.
            logprobs = _with_cache_work(tuple(item[1] for item in generated), cache)
            mx.eval(logprobs)
            result = {"tokens": tuple(item[0] for item in generated), "logprobs": logprobs}
            if include_state:
                result["state"] = _cache_observation(cache)
        finally:
            # A Python failure must not leave this trial's queued work running
            # during the next comparison, and the cache must not keep the
            # trial's arrays alive: the tracer would read them as retained.
            mx.synchronize(generation_stream)
            self._reset()
        return result

    def __call__(self, *inputs):
        return self._run(inputs)

    def correctness(self, *inputs):
        """Check generated outputs and all resulting cache state together."""
        return self._run(inputs, include_state=True)
