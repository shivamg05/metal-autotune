"""Small real Llama architecture for cached installation/export checks."""
import mlx.core as mx
import mlx.nn as nn


def build():
    from mlx_lm.models.llama import Model, ModelArgs

    mx.random.seed(19)
    model = Model(ModelArgs(
        model_type='llama', hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=64, rms_norm_eps=1e-6))
    model.eval()
    nn.quantize(model, group_size=64, bits=4)
    return model
