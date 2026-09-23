"""Exercise the Muse adapter with real MLX-VLM layers, without large downloads."""
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from models.muse_glimmer_30b_4bit import TextModel, build
from autotuner_runtime.state import context_step
from autotuner_runtime.inference import resolve_library_inference


def small_model():
    from mlx_vlm.models.muse_glimmer.config import TextConfig
    from mlx_vlm.models.muse_glimmer.language import LanguageModel

    model = LanguageModel(TextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, sliding_window=8,
    ))
    nn.quantize(model, bits=4, group_size=64)
    mx.eval(model.parameters())
    return model


def test_builder_keeps_language_model_and_logits():
    language = small_model()
    with patch('mlx_vlm.load', return_value=(SimpleNamespace(language_model=language), None)) as load:
        model = build()
    load.assert_called_once_with('mlx-community/Muse-Glimmer-30B-4bit', lazy=True)
    tokens = mx.array([[1, 2, 3]])
    assert model.language_model is language
    assert mx.array_equal(model(tokens), language(tokens).logits).item()
    assert resolve_library_inference(model, None, []) is False
    with pytest.raises(ValueError, match='not a complete MLX-LM'):
        resolve_library_inference(model, True, [])


@pytest.mark.parametrize('context', [0, 4, 10])
def test_cached_forward_is_repeatable_and_matches_library(context):
    language = small_model()
    model = TextModel(language)
    prefix = mx.ones((1, context), dtype=mx.int32)
    token = mx.array([[2]])
    cache = language.make_cache()
    if context:
        mx.eval(language(prefix, cache=cache).logits)
    expected = language(token, cache=cache).logits
    step = context_step(model, context, prefix, [token])
    first, second = step(token), step(token)
    assert mx.array_equal(first, expected).item()
    assert mx.array_equal(first, second).item()
    outputs = step.sequence([token], 3)
    assert len(outputs) == 3
    assert all(o.shape == (1, 1, 64) for o in outputs)
