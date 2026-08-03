"""List trained ONNX model files for the play server's model picker."""

import re
from pathlib import Path

_INT_RE = re.compile(r"\d+")


def _version_key(name: str) -> tuple[int, str]:
    ints = _INT_RE.findall(name)
    return (int(ints[-1]) if ints else -1, name)


def list_onnx_models(models_dir: Path) -> list[str]:
    """Basenames of ``*.onnx`` files in ``models_dir``, sorted by the trailing
    integer (so ``model_9`` precedes ``model_10``). Empty if the dir is absent."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    names = [p.name for p in models_dir.glob("*.onnx")]
    return sorted(names, key=_version_key)


def default_model(models: list[str]) -> str | None:
    """The highest-version model (last, given the natural sort), or None."""
    return models[-1] if models else None
