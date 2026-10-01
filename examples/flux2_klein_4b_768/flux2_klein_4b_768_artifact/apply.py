"""Load this artifact onto a freshly loaded build() model:

    from artifact.apply import apply
    model = apply(build())
"""

import sys
from pathlib import Path

_here = Path(__file__).parent
sys.path.insert(0, str(_here / "runtime"))

from autotuner_runtime.apply import apply as _apply


def apply(model):
    return _apply(model, _here)
