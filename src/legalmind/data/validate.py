from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.data.contracts import ProcessedCase


def load_split(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(ProcessedCase.model_validate_json(line).model_dump())
            except Exception as exc:
                raise ValueError(f"Invalid processed row at {path}:{line_number}: {exc}") from exc
    return rows


def validate_dataset(data_dir: str | Path) -> dict:
    source = Path(data_dir)
    mapping = json.loads((source / "label_mapping.json").read_text(encoding="utf-8"))
    valid_ids = set(mapping.values())
    splits = {name: load_split(source / f"{name}.jsonl") for name in ("train", "validation", "test")}
    fingerprints: dict[str, set[str]] = {}
    groups: dict[str, set[str]] = {}
    invalid_labels = 0
    false_review_claims = 0
    for name, rows in splits.items():
        fingerprints[name] = {row["cleaning_metadata"]["fact_sha256"] for row in rows}
        groups[name] = {row["cleaning_metadata"]["dedup_group_id"] for row in rows}
        invalid_labels += sum(not set(row["accusation_ids"]).issubset(valid_ids) for row in rows)
        false_review_claims += sum(row["review_status"] == "manually_reviewed" for row in rows)
    overlaps = {}
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps[f"{left}_{right}_facts"] = len(fingerprints[left] & fingerprints[right])
            overlaps[f"{left}_{right}_groups"] = len(groups[left] & groups[right])
    report = {
        "valid": invalid_labels == 0 and false_review_claims == 0 and not any(overlaps.values()),
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "num_labels": len(mapping),
        "invalid_label_rows": invalid_labels,
        "manually_reviewed_rows": false_review_claims,
        "overlaps": overlaps,
    }
    if not report["valid"]:
        raise ValueError(json.dumps(report, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_v2")
    parser.add_argument("--output", default="reports/data/validation_report.json")
    args = parser.parse_args()
    report = validate_dataset(args.data_dir)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
