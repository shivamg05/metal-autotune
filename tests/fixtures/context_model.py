"""A toy language model with mlx-lm's cache conventions: make_cache(), a
cache= keyword, and one cache per layer that grows its buffer in steps,
reports offset, and trims. Each layer sums every stored key, so the output
depends on how many tokens the context holds."""

import mlx.core as mx
import mlx.nn as nn

VOCAB, WIDTH, LAYERS, STEP = 32, 8, 2, 4


class Cache:
    """mlx-lm's KVCache in miniature: a buffer grown STEP rows at a time, a
    write position, and trim() to move the position back."""

    def __init__(self):
        self.keys = None
        self.offset = 0

    def update_and_fetch(self, keys):
        prev = self.offset
        if self.keys is None or prev + keys.shape[1] > self.keys.shape[1]:
            rows = ((keys.shape[1] + STEP - 1) // STEP) * STEP
            grown = mx.zeros((keys.shape[0], rows, keys.shape[2]), keys.dtype)
            self.keys = grown if self.keys is None else mx.concatenate([self.keys, grown], axis=1)
        self.offset += keys.shape[1]
        self.keys[:, prev:self.offset, :] = keys
        return self.keys[:, :self.offset, :]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wk = mx.full((WIDTH,), 0.5, dtype=mx.float32)
        self.wo = mx.full((WIDTH,), 2.0, dtype=mx.float32)

    def __call__(self, x, cache=None):
        keys = x * self.wk
        if cache is not None:
            keys = cache.update_and_fetch(keys)
        return (x + mx.sum(keys, axis=1, keepdims=True)) * self.wo


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, WIDTH)
        self.layers = [Layer() for _ in range(LAYERS)]

    def __call__(self, tokens, cache=None):
        x = self.embed(tokens)
        for i, layer in enumerate(self.layers):
            x = layer(x, None if cache is None else cache[i])
        return x

    def make_cache(self):
        return [Cache() for _ in self.layers]


def build():
    model = Model()
    model.embed.weight = mx.arange(VOCAB * WIDTH, dtype=mx.float32).reshape(VOCAB, WIDTH) / 64
    return model
