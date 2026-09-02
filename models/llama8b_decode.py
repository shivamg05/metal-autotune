"""Llama 3 8B Instruct 4-bit, wrapped as one repeatable decode step.

build() prefills a fixed 512-token context into the KV cache, then returns a
module whose every call runs the same single-token decode step against that
context: the cache offset is rewound after each call, so repeated calls are
bit-identical, which the harness's capture and pairing laws require. The
cache buffer is pre-grown inside build() (one warm call, then rewind) so the
first harness-visible call is structurally identical to every later one.
"""

import mlx.core as mx
import mlx.nn as nn

MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
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


def build():
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, _ = load(MODEL)
    cache = make_prompt_cache(model)
    prompt = mx.random.randint(0, 128000, (1, CONTEXT), key=mx.random.key(7))
    mx.eval(model(prompt, cache=cache))
    step = DecodeStep(model, cache, CONTEXT)
    warm = mx.random.randint(0, 128000, (1, 1), key=mx.random.key(8))
    mx.eval(step(warm))  # grows the cache buffer once so later calls never do
    return step
