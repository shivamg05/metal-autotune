"""Muse Glimmer 30B, pretrained 4-bit text model loaded through MLX-VLM.

Use use_library_inference: false. Forward calls return logits; context in the
manifest enables the model's own cache. Image inputs are not exposed here.
"""

import mlx.core as mx
import mlx.nn as nn


class TextModel(nn.Module):
    """Expose MLX-VLM's text model with the harness's array-output contract."""

    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    def __call__(self, tokens, cache=None):
        return self.language_model(tokens, cache=cache).logits

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()


def build():
    from mlx_vlm import load

    # Skip evaluating vision weights; only retain the pretrained text model.
    model, _ = load("mlx-community/Muse-Glimmer-30B-4bit", lazy=True)
    text = TextModel(model.language_model)
    mx.eval(text.parameters())
    return text
