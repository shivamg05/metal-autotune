"""Whisper small's pretrained fp16 encoder, taking [batch, frames, mel_bins]."""


def build():
    import mlx.core as mx
    from mlx_whisper.load_models import load_model

    return load_model("mlx-community/whisper-small-mlx", dtype=mx.float16).encoder
