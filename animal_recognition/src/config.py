"""Config loader — converts config.yaml into dot-accessible Python objects.

Usage:
    from src.config import load_config
    cfg = load_config()
    print(cfg.pipeline.classifier)   # 'baseline_cnn'
    print(cfg.training.lr)           # 0.001
"""
from __future__ import annotations

from pathlib import Path
import yaml

# Navigate from this file up to the project root where config.yaml lives:
# src/config.py → src/ → animal_recognition/ → project root
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class _NS:
    """Recursively converts a nested dict into dot-notation attributes.

    Example:
        ns = _NS({"a": {"b": 1}})
        ns.a.b  →  1
    """

    def __init__(self, mapping: dict):
        for key, value in mapping.items():
            setattr(self, key, _NS(value) if isinstance(value, dict) else value)

    def __repr__(self) -> str:
        return f"Config({vars(self)})"


def load_config(path: str | Path = _DEFAULT_PATH) -> _NS:
    """Load config.yaml and return as a nested namespace object.

    Args:
        path: Path to the YAML file. Defaults to <project_root>/config.yaml.

    Returns:
        Nested _NS object. Access fields with dot notation.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Make sure you are running from the project root or pass the correct path."
        )
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _NS(raw)
