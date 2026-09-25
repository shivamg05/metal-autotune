"""Whisper large-v3-turbo's pretrained fp16 encoder, taking [batch, frames, mel_bins].

Turbo keeps large-v3's full 32-layer encoder and shrinks only the decoder, so
the encoder is most of the work in a local transcription.
"""


def build():
    import mlx.core as mx
    from mlx_whisper.load_models import load_model

    return load_model("mlx-community/whisper-large-v3-turbo", dtype=mx.float16).encoder
