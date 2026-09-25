"""apply(): install a job's artifact onto a freshly loaded build() model.

Loads the swap table, the generated patch module, and the kernel specs, then
swaps wrapper instances in at their scope addresses. Works in a fresh process
with no harness import; a root-scope entry returns the wrapper in place of the
model, so callers use the return value.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from . import kernels as _kernels
from .swap import install


def apply(model: object, artifact_dir: str | Path) -> object:
    artifact_dir = Path(artifact_dir)
    table = json.loads((artifact_dir / "swap_table.json").read_text())
    metadata_path = artifact_dir / "bundle.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    from .state import ContextStep
    from .inference import LibraryInference
    # Recorded paths include the harness's ContextStep.model prefix. Normal
    # inference callers supply their own cache and pass the underlying model.
    bare_context_model = (metadata.get("context") is not None or metadata.get("use_library_inference")) \
        and not isinstance(model, (ContextStep, LibraryInference))

    spec_mod = importlib.util.spec_from_file_location(
        "autotune_patch_wrappers", artifact_dir / "patch" / "wrappers.py"
    )
    wrappers = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(wrappers)

    specs = {}
    for metal in sorted((artifact_dir / "kernels").glob("*.metal")):
        spec = _kernels.load_spec(metal)
        specs[spec.kernel_id] = spec

    patched = model
    for entry in table:
        cls = getattr(wrappers, entry["wrapper_class"])
        kernel_specs = {kid: specs[kid] for kid in entry["kernel_ids"]}
        path = entry["scope_path"]
        if bare_context_model:
            if path == "model":
                path = ""
            elif path.startswith("model."):
                path = path[len("model."):]
            else:
                raise ValueError(f"cached artifact contains a non-model patch path: {path!r}")
        if path == "":
            patched = cls(patched, kernel_specs)
        else:
            original = _resolve_child(patched, path)
            install(patched, path, cls(original, kernel_specs))
    return patched


def _resolve_child(model: object, path: str):
    from .swap import resolve

    _, _, child = resolve(model, path)
    return child
