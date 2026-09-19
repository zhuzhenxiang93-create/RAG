from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from legalmind.schemas import CaseChunk


_VALID_STATUSES = {"pending", "approved", "rejected"}


@dataclass(frozen=True)
class LabelOverride:
    case_id: str
    original_accusations: tuple[str, ...]
    corrected_accusations: tuple[str, ...]
    status: str
    reason_code: str


def load_label_overrides(path: str | Path) -> dict[str, LabelOverride]:
    overrides: dict[str, LabelOverride] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            status = str(row.get("status", "pending"))
            if status not in _VALID_STATUSES:
                raise ValueError(f"invalid override status at line {line_number}: {status}")
            case_id = str(row.get("case_id", "")).strip()
            original = tuple(dict.fromkeys(row.get("original_accusations") or []))
            corrected = tuple(dict.fromkeys(row.get("corrected_accusations") or []))
            if not case_id or not original or not corrected:
                raise ValueError(f"incomplete label override at line {line_number}")
            if case_id in overrides:
                raise ValueError(f"duplicate label override for case_id={case_id}")
            overrides[case_id] = LabelOverride(
                case_id=case_id,
                original_accusations=original,
                corrected_accusations=corrected,
                status=status,
                reason_code=str(row.get("reason_code", "manual_review")),
            )
    return overrides


def apply_label_overrides(
    chunks: list[CaseChunk], overrides: dict[str, LabelOverride]
) -> tuple[list[CaseChunk], dict[str, object]]:
    approved = {key: value for key, value in overrides.items() if value.status == "approved"}
    seen_cases: set[str] = set()
    mismatches: list[dict[str, object]] = []
    applied_chunks = 0
    output: list[CaseChunk] = []
    for chunk in chunks:
        override = approved.get(chunk.case_id)
        if override is None:
            output.append(chunk)
            continue
        seen_cases.add(chunk.case_id)
        if set(chunk.accusations) != set(override.original_accusations):
            mismatches.append(
                {
                    "case_id": chunk.case_id,
                    "expected": list(override.original_accusations),
                    "actual": list(chunk.accusations),
                }
            )
            output.append(chunk)
            continue
        output.append(chunk.model_copy(update={"accusations": list(override.corrected_accusations)}))
        applied_chunks += 1
    missing_case_ids = sorted(set(approved) - seen_cases)
    audit: dict[str, object] = {
        "configured": bool(overrides),
        "records": len(overrides),
        "pending": sum(item.status == "pending" for item in overrides.values()),
        "approved": len(approved),
        "rejected": sum(item.status == "rejected" for item in overrides.values()),
        "applied_chunks": applied_chunks,
        "mismatches": mismatches,
        "missing_case_ids": missing_case_ids,
    }
    if mismatches:
        raise ValueError(f"label override source-label mismatch: {mismatches[0]['case_id']}")
    return output, audit


__all__ = ["LabelOverride", "apply_label_overrides", "load_label_overrides"]
