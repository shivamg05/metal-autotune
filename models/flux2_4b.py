"""FLUX.2 transformer (~4B), one denoiser step, for architecture-only autotuning.

A faithful MLX port of the official FLUX.2 transformer (black-forest-labs/flux2,
the Klein4B config), flattened into one self-contained file with random weights,
because the autotuner optimizes the op graph, not the image: only the op
structure and shapes have to be real. build() constructs it, casts bf16, and
nn.quantize takes every projection to 4-bit quantized_matmul. One call is one
stateless denoiser forward over image + text tokens, so the harness can trace,
price, and time it. No checkpoint downloads.

Config (FLUX.2 Klein4B, ~3.9B): 5 double-stream blocks, 20 single-stream blocks,
hidden 3072 (24 heads x 128), mlp_ratio 3, image latent channels 128, text dim
7680, no guidance. Structure matches BFL: a fused qkv projection per stream in
the double blocks and a fused qkv+mlp projection in the single blocks, SwiGLU
feed-forward, modulation shared across blocks (not per block), timestep
conditioning only, no biases, interleaved 4-axis rope.
"""

import math

import mlx.core as mx
import mlx.nn as nn

HEADS = 24
HEAD_DIM = 128
HIDDEN = HEADS * HEAD_DIM          # 3072 (inner_dim)
MLP_RATIO = 3.0
MLP_INNER = int(HIDDEN * MLP_RATIO)  # 9216
DOUBLE_BLOCKS = 5
SINGLE_BLOCKS = 20
IN_CHANNELS = 128
TXT_DIM = 7680                     # joint_attention_dim
TIME_CHANNELS = 256
AXES_DIM = (32, 32, 32, 32)        # rope axes; sum//2 = HEAD_DIM//2 = 64
ROPE_THETA = 2000
BITS = 4
GROUP_SIZE = 64
WEIGHT_SEED = 42

# The manifest picks the token counts. A 1024x1024 image is 4096 image tokens
# (8x VAE, 2x2 patches) and the Klein pipeline pads prompts to 512 text tokens.
# The rope tables are built once for the longest sequence a manifest may
# declare and sliced per call.
MAX_TOKENS = 4608


def _timestep_embedding(t: mx.array, dim: int = TIME_CHANNELS) -> mx.array:
    t = 1000.0 * t.astype(mx.float32)  # BFL time_factor
    half = dim // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / half)
    args = t[:, None] * freqs[None]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def _rope_tables(seq: int):
    # cos/sin over a positional grid; exact ids do not change the op graph, only
    # the values, which the autotuner does not use
    pos0 = mx.arange(seq, dtype=mx.float32)[:, None]
    ids = mx.concatenate([pos0, mx.zeros((seq, len(AXES_DIM) - 1))], axis=1)
    cos_out, sin_out = [], []
    for i, dim in enumerate(AXES_DIM):
        omega = 1.0 / (ROPE_THETA ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        out = ids[:, i][:, None] * omega[None]
        cos_out.append(mx.cos(out))
        sin_out.append(mx.sin(out))
    return mx.concatenate(cos_out, axis=-1), mx.concatenate(sin_out, axis=-1)   # [seq, 64]


def _apply_rope(q, k, cos, sin):
    cos_b = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
    sin_b = sin.reshape(1, 1, sin.shape[0], sin.shape[1])

    def mix(x):
        xf = x.astype(mx.float32)
        x2 = xf.reshape(*xf.shape[:-1], -1, 2)
        real, imag = x2[..., 0], x2[..., 1]
        out = mx.stack([real * cos_b - imag * sin_b, imag * cos_b + real * sin_b], axis=-1)
        return out.reshape(x.shape).astype(x.dtype)

    return mix(q), mix(k)


def _heads(x):
    b, s, _ = x.shape
    return mx.transpose(mx.reshape(x, (b, s, HEADS, HEAD_DIM)), (0, 2, 1, 3))


def _merge(x):
    b = x.shape[0]
    return mx.reshape(mx.transpose(x, (0, 2, 1, 3)), (b, -1, HIDDEN))


class SwiGLU(nn.Module):
    def __call__(self, x):
        a, b = mx.split(x, 2, axis=-1)
        return nn.silu(a) * b


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_in = nn.Linear(HIDDEN, MLP_INNER * 2, bias=False)
        self.act = SwiGLU()
        self.linear_out = nn.Linear(MLP_INNER, HIDDEN, bias=False)

    def __call__(self, x):
        return self.linear_out(self.act(self.linear_in(x)))


class Modulation(nn.Module):
    """One shared modulation: silu(temb) -> params, grouped into `sets` triples
    of (shift, scale, gate). Shared across blocks, computed once per step."""

    def __init__(self, sets: int):
        super().__init__()
        self.sets = sets
        self.linear = nn.Linear(HIDDEN, HIDDEN * 3 * sets, bias=False)

    def __call__(self, temb):
        mod = self.linear(nn.silu(temb))[:, None, :]
        parts = mx.split(mod, 3 * self.sets, axis=-1)
        return tuple(parts[3 * i: 3 * (i + 1)] for i in range(self.sets))


class Attention(nn.Module):
    """Double-block joint attention (BFL SelfAttention x2): a fused qkv
    projection per stream, per-stream q/k RMSNorm, text-first concat, rope, one
    attention, split, separate output projections."""

    def __init__(self):
        super().__init__()
        self.img_qkv = nn.Linear(HIDDEN, HIDDEN * 3, bias=False)
        self.txt_qkv = nn.Linear(HIDDEN, HIDDEN * 3, bias=False)
        self.img_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.txt_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        for n in ("img_norm_q", "img_norm_k", "txt_norm_q", "txt_norm_k"):
            setattr(self, n, nn.RMSNorm(HEAD_DIM, eps=1e-5))

    def _qkv(self, x, qkv, nq, nk):
        q, k, v = mx.split(qkv(x), 3, axis=-1)
        q = nq(_heads(q).astype(mx.float32)).astype(x.dtype)
        k = nk(_heads(k).astype(mx.float32)).astype(x.dtype)
        return q, k, _heads(v)

    def __call__(self, img, txt, cos, sin):
        iq, ik, iv = self._qkv(img, self.img_qkv, self.img_norm_q, self.img_norm_k)
        tq, tk, tv = self._qkv(txt, self.txt_qkv, self.txt_norm_q, self.txt_norm_k)
        q, k = _apply_rope(mx.concatenate([tq, iq], axis=2),
                           mx.concatenate([tk, ik], axis=2), cos, sin)
        v = mx.concatenate([tv, iv], axis=2)
        o = _merge(mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM ** -0.5))
        t_o, i_o = o[:, :txt.shape[1]], o[:, txt.shape[1]:]
        return self.img_proj(i_o), self.txt_proj(t_o)


class ParallelSelfAttention(nn.Module):
    """Single-block attention and SwiGLU MLP from one fused projection."""

    def __init__(self):
        super().__init__()
        self.to_qkv_mlp_proj = nn.Linear(HIDDEN, HIDDEN * 3 + MLP_INNER * 2, bias=False)
        self.norm_q = nn.RMSNorm(HEAD_DIM, eps=1e-5)
        self.norm_k = nn.RMSNorm(HEAD_DIM, eps=1e-5)
        self.mlp_act = SwiGLU()
        self.to_out = nn.Linear(HIDDEN + MLP_INNER, HIDDEN, bias=False)

    def __call__(self, h, cos, sin):
        qkv, mlp = mx.split(self.to_qkv_mlp_proj(h), [HIDDEN * 3], axis=-1)
        qa, ka, va = mx.split(qkv, 3, axis=-1)
        q = self.norm_q(_heads(qa).astype(mx.float32)).astype(h.dtype)
        k = self.norm_k(_heads(ka).astype(mx.float32)).astype(h.dtype)
        q, k = _apply_rope(q, k, cos, sin)
        o = _merge(mx.fast.scaled_dot_product_attention(q, k, _heads(va), scale=HEAD_DIM ** -0.5))
        return self.to_out(mx.concatenate([o, self.mlp_act(mlp)], axis=-1))


class DoubleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(HIDDEN, eps=1e-6, affine=False)
        self.norm1_context = nn.LayerNorm(HIDDEN, eps=1e-6, affine=False)
        self.norm2 = nn.LayerNorm(HIDDEN, eps=1e-6, affine=False)
        self.norm2_context = nn.LayerNorm(HIDDEN, eps=1e-6, affine=False)
        self.attn = Attention()
        self.ff = FeedForward()
        self.ff_context = FeedForward()

    def __call__(self, img, txt, mod_img, mod_txt, cos, sin):
        (i_sa, i_sca, i_ga), (i_sm, i_scm, i_gm) = mod_img
        (t_sa, t_sca, t_ga), (t_sm, t_scm, t_gm) = mod_txt
        i_o, t_o = self.attn((1 + i_sca) * self.norm1(img) + i_sa,
                             (1 + t_sca) * self.norm1_context(txt) + t_sa, cos, sin)
        img = img + i_ga * i_o
        txt = txt + t_ga * t_o
        img = img + i_gm * self.ff((1 + i_scm) * self.norm2(img) + i_sm)
        txt = txt + t_gm * self.ff_context((1 + t_scm) * self.norm2_context(txt) + t_sm)
        return img, txt


class SingleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(HIDDEN, eps=1e-6, affine=False)
        self.attn = ParallelSelfAttention()

    def __call__(self, h, mod, cos, sin):
        shift, scale, gate = mod
        return h + gate * self.attn((1 + scale) * self.norm(h) + shift, cos, sin)


class TimestepEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(TIME_CHANNELS, HIDDEN, bias=False)
        self.linear_2 = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def __call__(self, timestep):
        return self.linear_2(nn.silu(self.linear_1(_timestep_embedding(timestep))))


class NormOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN * 2, bias=False)
        self.norm = nn.LayerNorm(HIDDEN, eps=1e-6, affine=False)

    def __call__(self, x, temb):
        scale, shift = mx.split(self.linear(nn.silu(temb)), 2, axis=-1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class Flux2(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(IN_CHANNELS, HIDDEN, bias=False)
        self.context_embedder = nn.Linear(TXT_DIM, HIDDEN, bias=False)
        self.time_embed = TimestepEmbed()
        self.mod_img = Modulation(sets=2)
        self.mod_txt = Modulation(sets=2)
        self.mod_single = Modulation(sets=1)
        self.double_blocks = [DoubleBlock() for _ in range(DOUBLE_BLOCKS)]
        self.single_blocks = [SingleBlock() for _ in range(SINGLE_BLOCKS)]
        self.norm_out = NormOut()
        self.proj_out = nn.Linear(HIDDEN, IN_CHANNELS, bias=False)
        cos, sin = _rope_tables(MAX_TOKENS)
        self._cos, self._sin = cos, sin

    def __call__(self, hidden_states, encoder_hidden_states, timestep):
        seq = encoder_hidden_states.shape[1] + hidden_states.shape[1]
        cos, sin = self._cos[:seq], self._sin[:seq]
        temb = self.time_embed(timestep).astype(hidden_states.dtype)
        img = self.x_embedder(hidden_states)
        txt = self.context_embedder(encoder_hidden_states)
        mod_img, mod_txt = self.mod_img(temb), self.mod_txt(temb)
        for blk in self.double_blocks:
            img, txt = blk(img, txt, mod_img, mod_txt, cos, sin)
        h = mx.concatenate([txt, img], axis=1)
        mod_single = self.mod_single(temb)[0]
        for blk in self.single_blocks:
            h = blk(h, mod_single, cos, sin)
        img = h[:, txt.shape[1]:]
        return self.proj_out(self.norm_out(img, temb))


def build():
    # apply() rebuilds this model in a fresh process. Architecture-only weights
    # still have to be the same weights for output reproduction to mean anything.
    mx.random.seed(WEIGHT_SEED)
    model = Flux2()
    model.apply(lambda a: a.astype(mx.bfloat16))
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    return model
