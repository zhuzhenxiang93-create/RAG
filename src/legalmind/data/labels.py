from __future__ import annotations

import ast
import json
from pathlib import Path


def load_label_mapping(path: str | Path) -> dict[str, int]:
    source = Path(path)
    raw = source.read_bytes()
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding)
            value = json.loads(text)
            return {str(key): int(item) for key, item in value.items()}
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    try:
        value = ast.literal_eval(raw.decode("utf-8", errors="replace"))
        return {str(key): int(item) for key, item in value.items()}
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Cannot parse label mapping {source}: {last_error or exc}") from exc


def save_label_mapping(mapping: dict[str, int], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(mapping.items(), key=lambda item: item[1]))
    target.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")


def build_label_mapping(label_sets: list[list[str]]) -> dict[str, int]:
    labels = sorted({label for values in label_sets for label in values})
    return {label: index for index, label in enumerate(labels)}
