"""Pretrained SD 2.1-base fp16 U-Net. Requires the official MLX Examples source.

Inputs: NHWC latent, timestep vector, text embeddings. See models/README.md.
"""


def build():
    try:
        from stable_diffusion.model_io import load_unet
    except ModuleNotFoundError as exc:
        if exc.name != "stable_diffusion":
            raise
        raise ImportError(
            "SD 2.1 needs the official MLX Examples stable_diffusion directory "
            "on PYTHONPATH. See models/README.md for setup."
        ) from exc

    return load_unet("stabilityai/stable-diffusion-2-1-base", float16=True)
