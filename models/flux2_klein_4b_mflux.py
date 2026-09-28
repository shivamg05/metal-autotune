"""FLUX.2 Klein 4B's denoiser exactly as mflux runs it, with the published weights.

build() returns mflux's own Flux2Transformer, taken from its Flux2Klein
pipeline, so a bundle from this model patches the same transformer mflux users
run. Using it is normal mflux plus one line:

    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein
    from artifact import apply

    pipe = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
    pipe.transformer = apply(pipe.transformer)
    pipe.generate_image(seed=42, prompt="a lighthouse at dusk", width=512, height=512).save("out.png")

The transformer is where generation spends its time: it runs once per
denoising step, while the text encoder and VAE run once per image. Needs
mflux (pip install mflux==0.20.0); weights download from Black Forest Labs'
Hugging Face repo on first use.
"""

QUANTIZE = 4  # bits; mflux's default is full precision (None), 8 is also common


def build():
    import mlx.core as mx
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein

    pipe = Flux2Klein(quantize=QUANTIZE, model_config=ModelConfig.flux2_klein_4b())
    transformer = pipe.transformer
    del pipe  # the text encoder and VAE are not part of the optimized call
    mx.clear_cache()
    return transformer
