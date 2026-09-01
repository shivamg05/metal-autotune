"""Llama 3 8B Instruct, 4-bit, exactly as mlx_lm loads it."""


def build():
    from mlx_lm import load
    model, _ = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    return model
