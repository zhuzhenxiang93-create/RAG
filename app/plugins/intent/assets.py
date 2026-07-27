"""Validation and loading for intent LoRA artifacts."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class IntentAssets:
    intents: Dict[str, int]
    intent_to_domain: Dict[str, str]
    base_model: str
    adapter_path: Path

    @property
    def intents_by_id(self) -> Dict[int, str]:
        return {identifier: label for label, identifier in self.intents.items()}


def load_taxonomy(path: Path) -> tuple[Dict[str, int], Dict[str, str]]:
    if not path.is_file():
        raise ValueError("Intent taxonomy file does not exist")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Intent taxonomy must be a JSON object")
    raw_intents = payload.get("intents")
    raw_domains = payload.get("intent_to_domain")
    if not isinstance(raw_intents, dict) or not raw_intents:
        raise ValueError("Intent taxonomy requires a non-empty 'intents' object")
    if not isinstance(raw_domains, dict):
        raise ValueError("Intent taxonomy requires an 'intent_to_domain' object")
    intents = {str(label): int(identifier) for label, identifier in raw_intents.items()}
    if sorted(intents.values()) != list(range(len(intents))):
        raise ValueError("Intent identifiers must be contiguous from 0")
    missing_domains = sorted(set(intents) - set(raw_domains))
    if missing_domains:
        raise ValueError("Missing domains for intents: {}".format(", ".join(missing_domains)))
    return intents, {intent: str(raw_domains[intent]) for intent in intents}


def validate_assets(
    *, base_model: str, adapter_path: str, labels_path: str
) -> IntentAssets:
    missing = [
        name
        for name, value in (
            ("base model", base_model),
            ("adapter path", adapter_path),
            ("taxonomy path", labels_path),
        )
        if not value
    ]
    if missing:
        raise ValueError("Missing intent plugin configuration: {}".format(", ".join(missing)))
    adapter = Path(adapter_path).expanduser()
    if not adapter.is_dir():
        raise ValueError("Intent adapter directory does not exist")
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError("Intent adapter_config.json is missing")
    intents, intent_to_domain = load_taxonomy(Path(labels_path).expanduser())
    return IntentAssets(
        intents=intents,
        intent_to_domain=intent_to_domain,
        base_model=base_model,
        adapter_path=adapter,
    )


def safe_label_count(labels_path: str) -> Optional[int]:
    if not labels_path:
        return None
    try:
        intents, _ = load_taxonomy(Path(labels_path).expanduser())
        return len(intents)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

