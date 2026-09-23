"""Qwen3.5-4B with pretrained 8-bit weights. See models/README.md.

The checkpoint is a vision-language model; mlx_lm loads only its language model.
"""


def build():
    from mlx_lm import load

    model, _ = load("mlx-community/Qwen3.5-4B-8bit")
    return model
