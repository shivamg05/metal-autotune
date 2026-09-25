"""load(): build the model from this bundle's own source and apply the patch.

Copied into each artifact as load.py. Imports nothing from the optimizer.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import mlx.core as mx

_HERE = Path(__file__).resolve().parent
_DEFINITION = None


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _definition(metadata):
    """Import model/<entry> once, under a private package so its relative and
    project-rooted imports resolve without the original repository."""
    global _DEFINITION
    if _DEFINITION is not None:
        return _DEFINITION
    source = _HERE / "model"
    entry = source / metadata["entry"]
    for directory in (source, entry.parent):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    name = "_artifact_model_" + hashlib.md5(str(_HERE).encode()).hexdigest()[:8]
    package = types.ModuleType(name)
    package.__path__ = [str(source)]
    sys.modules[name] = package
    relative = Path(metadata["entry"]).with_suffix("")
    for index in range(1, len(relative.parts)):
        parent_name = name + "." + ".".join(relative.parts[:index])
        folder = source.joinpath(*relative.parts[:index])
        init = folder / "__init__.py"
        if init.is_file():
            spec = importlib.util.spec_from_file_location(parent_name, init, submodule_search_locations=[str(folder)])
            parent = importlib.util.module_from_spec(spec)
            sys.modules[parent_name] = parent
            spec.loader.exec_module(parent)
        else:
            parent = types.ModuleType(parent_name)
            parent.__path__ = [str(folder)]
            sys.modules[parent_name] = parent
    spec = importlib.util.spec_from_file_location(name + "." + ".".join(relative.parts), entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # the optimizer imported the model as "autotune_model"; replays of compiled
    # helpers recorded by import path look it up under that name
    sys.modules["autotune_model"] = module
    spec.loader.exec_module(module)
    _DEFINITION = module
    return module


def _as_step(model, metadata, generated_tokens=None):
    """When the job's workload ran over a filled KV cache, the same step: the
    cache built the model's way, filled with the saved tokens, rewound after
    every call."""
    context = metadata.get("context")
    if metadata.get("use_library_inference"):
        from autotuner_runtime.inference import LibraryInference
        tokens = mx.load(str(_HERE / context["file"]))["tokens"] if context else None
        return LibraryInference(model, steps=(metadata["final_benchmark"]["steps"]
                                              if generated_tokens is None else generated_tokens),
                                prefix_tokens=tokens)
    if not context:
        return model
    sys.path.insert(0, str(_HERE / "runtime"))
    from autotuner_runtime.state import context_step

    tokens = mx.load(str(_HERE / context["file"]))["tokens"]
    warm = mx.load(str(_HERE / metadata["workloads"][context["workload"]]["file"]))
    return context_step(model, context["context"], tokens, [warm[f"i{k}"] for k in range(len(warm))])


class LoadedModel:
    """Callable forward plus the model object and its source module."""

    def __init__(self, model, definition, compiled):
        self.model = model
        self.definition = definition
        self.compiled = compiled
        self._forward = mx.compile(lambda *args: model(*args)) if compiled else model

    @property
    def inference_model(self):
        """The patched model for normal inference with a caller-owned cache.

        The bundle callable itself retains the job's repeatable benchmark
        step. The underlying model advances the cache normally.
        """
        metadata = json.loads((_HERE / "bundle.json").read_text())
        return self.model.model if metadata.get("context") or metadata.get("use_library_inference") else self.model

    def __call__(self, *inputs):
        return self._forward(*inputs)


def load(*, patched=True, compile=None, share_weights_with=None, generated_tokens=None,
         measurement_baseline=False):
    """Build the model from model/, apply the patch, run it as the job measured it.

    patched=False gives the untouched model built the same way. Set
    measurement_baseline=True to restore the original compiled scopes for timing.
    compile defaults to the job's baseline: the forward pass runs under mx.compile
    when the job measured against the compiled model. share_weights_with
    takes a model built earlier from this bundle; the new model then points
    at that model's weight arrays instead of holding a second copy, which is
    how the job kept two models resident for its comparisons.
    """
    metadata = json.loads((_HERE / "bundle.json").read_text())
    if measurement_baseline and patched:
        raise ValueError("measurement_baseline requires patched=False")
    if (measurement_baseline and metadata.get("use_library_inference")
            and metadata["baseline"] == "compiled" and not metadata.get("baseline_scopes")):
        raise ValueError("bundle lacks the compiled baseline scopes; re-export it before benchmarking")
    if compile and (metadata.get("context") or metadata.get("use_library_inference")):
        raise ValueError("the inference/cache controller cannot be compiled from outside the model")
    if generated_tokens is not None and not metadata.get("use_library_inference"):
        raise ValueError("generated_tokens requires a library inference bundle")
    sys.path.insert(0, str(_HERE / "runtime"))
    from autotuner_runtime.checkpoints import bundle_checkpoint_pins, use_checkpoint_pins
    with use_checkpoint_pins(bundle_checkpoint_pins(metadata, _HERE)):
        module = _definition(metadata)
        model = module.build()
    if share_weights_with is not None:
        from autotuner_runtime.swap import require_independent_models
        from autotuner_runtime.state import ContextStep
        from autotuner_runtime.inference import LibraryInference
        if isinstance(share_weights_with, (ContextStep, LibraryInference)):
            share_weights_with = share_weights_with.model
        require_independent_models(model, share_weights_with)
        if hasattr(model, "update"):
            model.update(share_weights_with.parameters())
    # Fill the context only after both arms have the same weights.
    model = _as_step(model, metadata, generated_tokens)
    if hasattr(model, "parameters"):
        mx.eval(model.parameters())
    if patched:
        model = _module("_artifact_apply", _HERE / "apply.py").apply(model)
    if measurement_baseline and metadata.get("baseline_scopes"):
        from autotuner_runtime.apply import apply
        model = apply(model, _HERE, baseline=True)
    compiled = metadata["baseline"] == "compiled" if compile is None else bool(compile)
    if metadata.get("use_library_inference"):
        compiled = False
    return LoadedModel(model, module, compiled)
