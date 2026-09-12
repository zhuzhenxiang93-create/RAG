from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import date


def temporal_group_split(rows: list[dict], validation_ratio: float = 0.1, test_ratio: float = 0.1) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_case_group_id"]].append(row)
    dated, undated = [], []
    for group_id, members in groups.items():
        dates = [date.fromisoformat(str(r["judgment_date"])) for r in members if r.get("judgment_date")]
        (dated if dates else undated).append((min(dates) if dates else None, group_id, members))
    dated.sort(key=lambda item: (item[0], item[1]))
    n = len(dated)
    test_start = int(n * (1 - test_ratio))
    val_start = int(n * (1 - test_ratio - validation_ratio))
    out = {"train": [], "validation": [], "test": []}
    for index, (_, _, members) in enumerate(dated):
        split = "train" if index < val_start else "validation" if index < test_start else "test"
        out[split].extend(members)
    for _, group_id, members in undated:
        bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 10_000 / 10_000
        split = "test" if bucket < test_ratio else "validation" if bucket < test_ratio + validation_ratio else "train"
        out[split].extend(members)
    return out
