"""MetalBench standard/silu_residual: build() wraps the vendored Model so one call is one forward."""
import importlib.util
from pathlib import Path

import mlx.nn as nn

PROBLEM = Path('<repo>/metalbench/problems/standard/silu_residual.py')


def build():
    spec = importlib.util.spec_from_file_location("metalbench_standard_silu_residual", PROBLEM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Model(module.Model):  # the vendored forward, reachable as a swappable child scope
        def __call__(self, *inputs):
            return self.forward(*inputs)

    class Problem(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Model()

        def __call__(self, *inputs):
            return self.model(*inputs)

    return Problem()
