"""Small MLX model for learning the tool, with random weights and no downloads."""
import mlx.nn as nn


def build():
    return nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
