"""MetalBench standard/mul_gelu: build() wraps the vendored Model so one call is one forward."""
import importlib.util
from pathlib import Path

import mlx.nn as nn

PROBLEM = Path('<repo>/metalbench/problems/standard/mul_gelu.py')


def build():
    spec = importlib.util.spec_from_file_location("metalbench_standard_mul_gelu", PROBLEM)
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
