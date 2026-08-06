"""Shared configuration and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load YAML mapping and return it with absolute config parent directory."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return document, path.parent


def resolve_path(value: str | Path, base_dir: str | Path) -> Path:
    """Resolve relative path against explicit base directory."""
    path = Path(value)
    return path if path.is_absolute() else Path(base_dir) / path
