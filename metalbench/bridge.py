"""MetalBench problems as autotune jobs.

A problem is one vendored MLX module (`class Model` with `forward(*inputs)`)
plus the input shapes its registry lists; weights arrive as inputs. The bridge
writes, per problem, a model file whose build() wraps that Model and a
manifest naming its shapes, tolerances and budget. The harness does the rest:
the compiled model is the baseline, the ladder decides correctness, and the
report carries both step clocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBLEMS = HERE / "problems"
GENERATED = HERE / "generated"
SETS = ("common", "standard", "full")

MODEL_FILE = '''"""MetalBench {set}/{name}: build() wraps the vendored Model so one call is one forward."""
import importlib.util
from pathlib import Path

import mlx.nn as nn

PROBLEM = Path({module!r})


def build():
    spec = importlib.util.spec_from_file_location("metalbench_{set}_{name}", PROBLEM)
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
'''


@dataclass(frozen=True)
class Problem:
    set: str
    name: str
    input_shapes: tuple[tuple[int, ...], ...]
    rtol: float
    atol: float

    @property
    def module(self) -> Path:
        return PROBLEMS / self.set / f"{self.name}.py"


def problems(sets: tuple[str, ...] = SETS) -> dict[str, Problem]:
    """Every registered problem, first set wins a duplicated name."""
    found: dict[str, Problem] = {}
    for set_name in sets:
        namespace: dict = {}
        exec((PROBLEMS / set_name / "registry.py").read_text(), namespace)
        for name, entry in namespace["REGISTRY"].items():
            if name in found or not (PROBLEMS / set_name / f"{name}.py").exists():
                continue
            found[name] = Problem(set_name, name, tuple(tuple(s) for s in entry["input_shapes"]),
                                  float(entry["rtol"]), float(entry["atol"]))
    return found


def write_job(problem: Problem, budget_per_region: int, budget_total: int,
              out: Path = GENERATED) -> Path:
    """The model file and manifest for one problem; returns the manifest path."""
    job = out / problem.set / problem.name
    job.mkdir(parents=True, exist_ok=True)
    (job / "model.py").write_text(MODEL_FILE.format(set=problem.set, name=problem.name,
                                                    module=str(problem.module)))
    inputs = "".join(f"      - shape: {list(shape)}\n        dtype: float32\n" for shape in problem.input_shapes)
    (job / "manifest.yaml").write_text(
        f"model: model.py\nworkloads:\n  - name: {problem.name}\n    inputs:\n{inputs}"
        f"tolerances: {{rtol: {problem.rtol}, atol: {problem.atol}}}\n"
        f"budget: {{per_region: {budget_per_region}, total: {budget_total}}}\n")
    return job / "manifest.yaml"
