"""MLX graph rewriting, compiled once during setup and reusable in exported bundles.

``rewrite(roots, pattern_outputs, parameters, replacement)`` returns new roots
and the number of replacements. The callback receives ordered boundary inputs
and returns ordered boundary outputs, or ``None`` to retain that occurrence.
All calls are graph construction work; cache/compile their result for inference.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile

_module = None
_name = "_graph_native"


def _identity() -> dict:
    root = Path(__file__).parent
    source = b"".join((root / name).read_bytes() for name in ("graph_native.cpp", "CMakeLists.txt"))
    version = importlib.metadata.version("mlx")
    if version != "0.32.2":
        raise RuntimeError(f"Graph installation supports MLX 0.32.2; found {version}. The native API must be validated before upgrading.")
    return {
        "mlx": version,
        "python": sys.implementation.cache_tag,
        "platform": sysconfig.get_platform(),
        "machine": platform.machine(),
        "source": hashlib.sha256(source).hexdigest(),
    }


def _build(directory: Path) -> Path:
    import mlx.core as mx

    try:
        import cmake
        import nanobind
    except ImportError as exc:
        raise RuntimeError("Graph installation needs the package build dependencies: nanobind==2.15.0 and cmake>=3.27") from exc
    if importlib.metadata.version("nanobind") != "2.15.0":
        raise RuntimeError("Graph installation requires nanobind==2.15.0 to match MLX's Python array bindings")
    root = Path(__file__).parent.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    executable = str(Path(cmake.CMAKE_BIN_DIR) / "cmake")
    with tempfile.TemporaryDirectory(prefix="graph-build-", dir=directory) as temporary:
        build = Path(temporary)
        commands = [
            [executable, "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
             f"-DPython_EXECUTABLE={sys.executable}", f"-Dnanobind_DIR={nanobind.cmake_dir()}",
             f"-DMLX_DIR={Path(mx.__file__).parent / 'share/cmake/MLX'}"],
            [executable, "--build", str(build), "-j", "2"],
        ]
        for command in commands:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError("Could not build the MLX graph extension:\n" + result.stdout[-8000:])
        built = build / (_name + sysconfig.get_config_var("EXT_SUFFIX"))
        output = directory / built.name
        # Atomic publication also allows independent worker processes to prepare it.
        os.replace(built, output)
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stamp:
        json.dump(_identity(), stamp, sort_keys=True)
    os.replace(stamp.name, directory / "build.json")
    return output


def _compatible(stamp: Path, identity: dict) -> bool:
    try:
        return json.loads(stamp.read_text()) == identity
    except (OSError, ValueError):
        return False


def prepare(destination: str | Path | None = None) -> Path:
    """Prepare the extension, optionally copying its binary and ABI stamp for export."""
    identity = _identity()
    filename = _name + sysconfig.get_config_var("EXT_SUFFIX")
    packaged = Path(__file__).parent / filename
    stamp = packaged.with_name("build.json")
    if packaged.is_file():
        if not _compatible(stamp, identity):
            raise RuntimeError("The bundled graph extension does not match this Python/MLX/platform version; rebuild the artifact runtime")
        path = packaged
    else:
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "metal-autotune" / "graph" / digest
        path = cache / filename
        stamp = cache / "build.json"
        if not path.is_file() or not _compatible(stamp, identity):
            path = _build(cache)
    if destination is not None:
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        output = target / filename
        if path.resolve() != output.resolve():
            shutil.copy2(path, output)
            shutil.copy2(path.with_name("build.json"), target / "build.json")
        return output
    return path


def _load():
    global _module
    if _module is None:
        import mlx.core  # Register MLX's array type and load its shared library first.

        path = prepare()
        spec = importlib.util.spec_from_file_location(_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load graph extension at {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _module = module
    return _module


def rewrite(roots, pattern_outputs, parameters, replacement, *, anchors=()):
    """Known boundary arrays constrain matching before choosing an occurrence."""
    return _load().rewrite(roots, pattern_outputs, parameters, replacement, anchors)


def same_structure(original_roots, rewritten_roots):
    return _load().same_structure(original_roots, rewritten_roots)


def array_id(array):
    return _load().array_id(array)
