from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping: {config_path}")
    return data


def project_path(value: str | Path, root: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base = Path(root) if root else Path.cwd()
    return (base / path).resolve()
