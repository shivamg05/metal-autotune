"""Fixture: a LLaMA-shaped stack at toy scale. Embedding gather, rms_norm,
rope, causal sdpa, SwiGLU MLP, residual streams, tied final projection: the
op surface a real decoder exercises, with 8 identical layers for copy
grouping."""

import mlx.core as mx
import mlx.nn as nn

DIM, HEADS, LAYERS, VOCAB = 256, 4, 8, 512


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = nn.RMSNorm(DIM)
        self.wq = nn.Linear(DIM, DIM, bias=False)
        self.wk = nn.Linear(DIM, DIM, bias=False)
        self.wv = nn.Linear(DIM, DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)
        self.mlp_norm = nn.RMSNorm(DIM)
        self.gate = nn.Linear(DIM, 4 * DIM, bias=False)
        self.up = nn.Linear(DIM, 4 * DIM, bias=False)
        self.down = nn.Linear(4 * DIM, DIM, bias=False)

    def __call__(self, x):
        b, l, _ = x.shape
        h = self.attn_norm(x)
        q = self.wq(h).reshape(b, l, HEADS, -1).transpose(0, 2, 1, 3)
        k = self.wk(h).reshape(b, l, HEADS, -1).transpose(0, 2, 1, 3)
        v = self.wv(h).reshape(b, l, HEADS, -1).transpose(0, 2, 1, 3)
        q = mx.fast.rope(q, DIM // HEADS, traditional=False, base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, DIM // HEADS, traditional=False, base=10000.0, scale=1.0, offset=0)
        att = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=(DIM // HEADS) ** -0.5, mask="causal"
        )
        att = att.transpose(0, 2, 1, 3).reshape(b, l, DIM)
        x = x + self.wo(att)
        h = self.mlp_norm(x)
        return x + self.down(nn.silu(self.gate(h)) * self.up(h))


class LlamaIsh(nn.Module):
    def __init__(self):
        super().__init__()
        mx.random.seed(23)
        self.embed = nn.Embedding(VOCAB, DIM)
        self.layers = [Block() for _ in range(LAYERS)]
        self.final_norm = nn.RMSNorm(DIM)

    def __call__(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x @ self.embed.weight.T


def build():
    return LlamaIsh()
