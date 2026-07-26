"""Validation of legal plugin labels and model artifacts."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class LegalAssets:
    labels: Dict[str, int]
    base_model: str
    adapter_path: Path
    score_weights_path: Path

    @property
    def labels_by_id(self) -> Dict[int, str]:
        return {identifier: label for label, identifier in self.labels.items()}


def load_labels(path: Path) -> Dict[str, int]:
    if not path.is_file():
        raise ValueError("Legal labels file does not exist")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Legal labels must be a non-empty JSON object")
    labels = {str(label): int(identifier) for label, identifier in payload.items()}
    identifiers = sorted(labels.values())
    if identifiers != list(range(len(labels))):
        raise ValueError("Legal label identifiers must be contiguous from 0")
    return labels


def validate_assets(
    *,
    base_model: str,
    adapter_path: str,
    score_weights_path: str,
    labels_path: str,
) -> LegalAssets:
    missing = [
        name
        for name, value in (
            ("base model", base_model),
            ("adapter path", adapter_path),
            ("score weights path", score_weights_path),
            ("labels path", labels_path),
        )
        if not value
    ]
    if missing:
        raise ValueError("Missing legal plugin configuration: {}".format(", ".join(missing)))

    adapter = Path(adapter_path).expanduser()
    score_weights = Path(score_weights_path).expanduser()
    labels_file = Path(labels_path).expanduser()
    if not adapter.is_dir():
        raise ValueError("Legal adapter directory does not exist")
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError("Legal adapter_config.json is missing")
    if not score_weights.is_file():
        raise ValueError("Legal classification-head weights do not exist")
    labels = load_labels(labels_file)
    return LegalAssets(
        labels=labels,
        base_model=base_model,
        adapter_path=adapter,
        score_weights_path=score_weights,
    )


def safe_label_count(labels_path: str) -> Optional[int]:
    if not labels_path:
        return None
    try:
        return len(load_labels(Path(labels_path).expanduser()))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
