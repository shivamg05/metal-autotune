"""Fixture: a step that keeps state in Python, the way a KV cache does. Each
call writes one slot of a buffer the module holds and reads the buffer back,
always the same slot, so repeated calls are bit-identical. mx.compile cannot
swap state it is not handed in a dict or list, so a compiled call would leave
the buffer holding a tracer and kill the model; the harness must give this
step the plain baseline."""

import types

import mlx.core as mx
import mlx.nn as nn


class KVStep(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(23)
        self.w = mx.random.normal((16, 16))
        self.cache = types.SimpleNamespace(buf=mx.zeros((4, 16)))  # held as an attribute, like a KV cache

    def __call__(self, x):
        k = x @ self.w
        self.cache.buf[0:1] = k[0:1]
        return k + self.cache.buf.sum(axis=0)


def build():
    return KVStep()
