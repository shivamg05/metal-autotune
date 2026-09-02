"""Fixture: a step whose cache is an object with a method, the way mlx_lm's
KV cache is. The write to the cache happens inside the method, so the
recorder sees one state call rather than a slice write it could never
replay, and the chains before and after it can be delivered at the layer
scope: its wrapper calls the same method on the same object. The root
rewinds the offset after every call so the step is repeatable."""

import mlx.core as mx
import mlx.nn as nn


class Cache:
    def __init__(self):
        self.keys = mx.zeros((8, 16))
        self.offset = 4

    def update_and_fetch(self, k):
        self.keys[self.offset:self.offset + 1] = k[0:1]
        self.offset += 1
        return self.keys[:self.offset]


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(31)
        self.g = mx.random.normal((16,))
        self.b = mx.random.normal((16,))

    def __call__(self, x, cache):
        k = mx.tanh((x + self.g) * self.b)  # a chain one kernel can replace, before the state call
        kv = cache.update_and_fetch(k)     # the state call: a hidden write with real side effects
        return kv.sum(axis=0) + x          # a chain after it


class Step(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Layer()
        self.cache = Cache()

    def __call__(self, x):
        y = self.layer(x, self.cache)
        self.cache.offset = 4              # rewind: every call is the same step
        return y


def build():
    return Step()
