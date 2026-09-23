"""LFM2.5-2.6B with pretrained 4-bit weights. See models/README.md."""


def build():
    from mlx_lm import load

    model, _ = load("LiquidAI/LFM2.5-2.6B-MLX-4bit")
    return model
