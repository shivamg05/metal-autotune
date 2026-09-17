"""A toy decode model that reads its KV cache's offset and feeds it to
mx.fast.rope, the way mlx-lm's attention does. The recorded step sees a fixed
offset; a correct wrapper must replay rope at the cache's live offset, so the
model stays right as generation advances past the recorded position."""

import mlx.core as mx
import mlx.nn as nn

VOCAB, HEADS, HEAD_DIM, LAYERS, STEP = 16, 1, 8, 1, 4
DIM = HEADS * HEAD_DIM


class Cache:
    """mlx-lm's KVCache in miniature: a buffer grown STEP rows at a time, an
    offset (tokens so far), and trim() to move it back."""

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
        self.w = mx.full((DIM,), 1.5, dtype=mx.float32)

    def __call__(self, x, cache=None):
        B, L, _ = x.shape
        q = x.reshape(B, L, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        q = mx.fast.rope(q, HEAD_DIM, traditional=False, base=10000.0, scale=1.0, offset=offset)
        qf = q.transpose(0, 2, 1, 3).reshape(B, L, DIM)
        ctx = cache.update_and_fetch(qf) if cache is not None else qf
        return (qf + mx.sum(ctx, axis=1, keepdims=True)) * self.w


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
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
    model.embed.weight = mx.arange(VOCAB * DIM, dtype=mx.float32).reshape(VOCAB, DIM) / 32
    return model
