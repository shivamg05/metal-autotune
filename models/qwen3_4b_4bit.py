"""Qwen3-4B with pretrained 4-bit weights. See models/README.md."""


def build():
    from mlx_lm import load

    model, _ = load("mlx-community/Qwen3-4B-4bit")
    return model
