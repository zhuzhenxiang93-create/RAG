from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from legalmind.data.normalize import normalize_record
from legalmind.schemas import CaseRecord


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {source}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"Expected object at {source}:{line_number}")
            yield value


def iter_cases(
    path: str | Path,
    source_split: str,
    label_to_id: dict[str, int] | None = None,
) -> Iterator[CaseRecord]:
    for raw in iter_jsonl(path):
        yield normalize_record(raw, source_split=source_split, label_to_id=label_to_id)
