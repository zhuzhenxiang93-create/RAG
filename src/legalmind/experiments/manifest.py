from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_snapshot() -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        values.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        values["torch"] = None
    return values


def source_revision(root: str | Path = ".") -> str:
    root = Path(root)
    if (root / ".git").exists():
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    marker = root / ".source-commit"
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else "unversioned"


def write_run_manifest(output_dir: str | Path, values: dict[str, Any]) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "created",
        "source_revision": source_revision(),
        "environment": environment_snapshot(),
        **values,
    }
    path = target / "run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
