"""Pretrained RecurrentGemma 2B. Prefill only; requires Hugging Face access."""


def build():
    from mlx_lm import load

    model, _ = load("google/recurrentgemma-2b")
    return model
