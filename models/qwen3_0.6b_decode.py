"""Qwen3 0.6B Base in bf16, exactly as mlx_lm loads it, wrapped as one
repeatable decode step.

build() prefills a fixed 512-token context into the KV cache, then returns a
module whose every call runs the same single-token decode step against that
context: the cache offset is rewound after each call, so repeated calls are
bit-identical, which the harness's capture and pairing laws require. The
cache buffer is pre-grown inside build() (one warm call, then rewind) so the
first harness-visible call is structurally identical to every later one.

The weights download from Hugging Face on the first build (about 1.2 GB).
"""

import mlx.core as mx
import mlx.nn as nn

MODEL = "Qwen/Qwen3-0.6B-Base"
VOCAB = 151936   # the published config's vocab_size; the manifest samples token ids below it
CONTEXT = 512


class DecodeStep(nn.Module):
    def __init__(self, inner, cache, offset):
        super().__init__()
        self.inner = inner
        self._cache = cache
        self._offset = offset

    def __call__(self, tok):
        out = self.inner(tok, cache=self._cache)
        for c in self._cache:
            c.offset = self._offset  # rewind: every call is the same step
        return out


def wrap(model, cache, context: int):
    """One repeatable decode step over a cache already holding the context."""
    step = DecodeStep(model, cache, context)
    warm = mx.random.randint(0, VOCAB, (1, 1), key=mx.random.key(8))
    mx.eval(step(warm))  # grows the cache buffer once so later calls never do
    return step


def build():
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, _ = load(MODEL)
    cache = make_prompt_cache(model)
    prompt = mx.random.randint(0, VOCAB, (1, CONTEXT), key=mx.random.key(7))
    mx.eval(model(prompt, cache=cache))
    return wrap(model, cache, CONTEXT)
