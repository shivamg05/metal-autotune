"""Qwen3 0.6B Base in bf16, exactly as mlx_lm loads it. The manifest decides
what one call is: a prompt, or one token over a conversation already in place."""


def build():
    from mlx_lm import load

    model, _ = load("Qwen/Qwen3-0.6B-Base")
    return model
