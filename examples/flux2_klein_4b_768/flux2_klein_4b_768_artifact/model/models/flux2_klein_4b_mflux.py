"""FLUX.2 Klein 4B's denoiser as mflux runs it, with random weights.

build() constructs mflux's own Flux2Transformer with Klein 4B's configuration
and quantizes it the way mflux does (every layer that can be, 4-bit, group
size 64, bfloat16 elsewhere), so a bundle from this model patches the same
transformer class mflux users run. The weights are random: speed depends on
the model's operations and shapes, not on the values, so nothing is
downloaded. Check an optimized bundle on the published weights before
relying on its outputs.

One call is one denoising step: mflux calls the transformer once per step
with the image tokens, the prompt's text features (always padded to 512
tokens), the timestep and both tokens' position ids. On an M3 or newer mflux
compiles that step with mx.compile, so the compiled model is the bar.

Using a bundle is normal mflux plus one line:

    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein
    from artifact import apply

    pipe = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
    pipe.transformer = apply(pipe.transformer)
    pipe.generate_image(seed=42, prompt="a lighthouse at dusk", width=768, height=768).save("out.png")

Needs mflux (pip install mflux==0.20.0).
"""

QUANTIZE = 4       # bits, as `Flux2Klein(quantize=4)`; mflux's default is full precision
WEIGHT_SEED = 7    # the same random weights in every process, so exported bundles reproduce


def build():
    import mlx.core as mx
    import mlx.nn as nn
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer

    mx.random.seed(WEIGHT_SEED)
    transformer = Flux2Transformer(**ModelConfig.flux2_klein_4b().transformer_overrides)
    transformer.apply(lambda a: a.astype(ModelConfig.precision))
    nn.quantize(transformer, group_size=64, bits=QUANTIZE,
                class_predicate=lambda path, module: hasattr(module, "to_quantized"))
    return transformer
