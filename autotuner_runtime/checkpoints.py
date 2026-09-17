"""Use pinned MLX-LM checkpoint paths in rebuilds and standalone bundles."""
from contextlib import contextmanager
import functools
from pathlib import Path


def bundle_checkpoint_pins(metadata, directory):
    """Prefer bundled copies; recovery checkpoints retain pinned cache paths."""
    if metadata.get('weight_policy') == 'model_source':
        return []
    pins = metadata.get('checkpoints') or metadata.get('checkpoint_pins') or []
    result = []
    for pin in pins:
        row = dict(pin)
        if row.get('directory'):
            root = Path(directory).resolve()
            target = (root / row['directory']).resolve()
            if not target.is_relative_to(root):
                raise ValueError('bundled checkpoint path escapes its artifact')
            row['path'] = str(target)
        result.append(row)
    return result


@contextmanager
def use_checkpoint_pins(pins):
    """Rebuild later model arms from the same files loaded before search."""
    if not pins:
        yield
        return
    import mlx_lm.utils as utils
    original = utils._download
    by_key = {(pin['source'], pin.get('requested_revision')): pin for pin in pins}
    sources = {pin['source'] for pin in pins}

    @functools.wraps(original)
    def download(path_or_hf_repo, revision=None, allow_patterns=None):
        source = str(path_or_hf_repo)
        pin = by_key.get((source, revision))
        if pin is None:
            if source in sources:
                raise ValueError(f'checkpoint revision changed after preflight: {source}')
            return original(path_or_hf_repo, revision=revision, allow_patterns=allow_patterns)
        path = Path(pin['path'])
        if not path.is_dir():
            raise ValueError(f'pinned checkpoint is no longer available: {path}')
        return original(path, revision=None, allow_patterns=allow_patterns)

    utils._download = download
    try:
        yield
    finally:
        utils._download = original
