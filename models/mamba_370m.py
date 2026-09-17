"""Pretrained Mamba-370M in its checkpoint FP16 precision. See models/README.md."""


def build():
    from mlx_lm import load

    model, _ = load("mlx-community/mamba-370m-hf-f16")
    return model
