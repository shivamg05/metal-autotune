"""Small real Qwen3.5 architecture, with both recurrence and attention caches."""
import mlx.core as mx
import mlx.nn as nn


def build():
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    mx.random.seed(19)
    model = TextModel(TextModelArgs(
        model_type='qwen3_5', hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=64, head_dim=16, full_attention_interval=2,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32))
    model.eval()
    nn.quantize(model, group_size=64, bits=4)
    return model
