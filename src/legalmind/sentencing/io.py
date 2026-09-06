from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows
