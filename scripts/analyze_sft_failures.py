from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from legalmind.generation.structured import (
    extract_json_object,
    parse_and_validate_json,
    parse_simplified_json,
)


def analyze(path: Path) -> tuple[list[dict], dict]:
    rows = []
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            output = row["output"]
            reasons = []
            try:
                payload = extract_json_object(output)
                parsed = json.loads(payload)
                counts["valid_json"] += 1
                if any(any("\u4e00" <= char <= "\u9fff" for char in key) for key in parsed):
                    reasons.append("chinese_field_names")
                if output.strip() != payload:
                    reasons.append("extra_text_outside_json")
            except ValueError as error:
                parsed = None
                reasons.append("truncated_or_missing_json")
                counts[str(error)] += 1
            except json.JSONDecodeError:
                parsed = None
                reasons.append("invalid_json_syntax")
            try:
                parse_and_validate_json(output)
            except (ValueError, TypeError, json.JSONDecodeError, ValidationError):
                reasons.append("full_schema_mismatch")
            try:
                parse_simplified_json(output)
            except (ValueError, TypeError, json.JSONDecodeError, ValidationError):
                reasons.append("simplified_schema_mismatch")
            try:
                parse_simplified_json(output, repair=True)
                counts["canonical_repair_success"] += 1
            except (ValueError, TypeError, json.JSONDecodeError, ValidationError):
                reasons.append("canonical_repair_failed")
            if len(row.get("raw_output", "")) >= 350:
                reasons.append("near_generation_token_limit")
            for reason in set(reasons):
                counts[reason] += 1
            rows.append(
                {
                    "case_id": row.get("case_id"),
                    "failure_reasons": sorted(set(reasons)),
                    "output_characters": len(output),
                    "reference_accusations": row.get("reference_accusations", []),
                    "parsed_keys": sorted(parsed) if isinstance(parsed, dict) else [],
                }
            )
    return rows, {"rows": len(rows), "failure_counts": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", required=True)
    parser.add_argument("--output", default="reports/generation/sft_smoke_failure_analysis.json")
    args = parser.parse_args()
    combined = {}
    for value in args.predictions:
        path = Path(value)
        rows, summary = analyze(path)
        combined[path.parent.name] = {"summary": summary, "rows": rows}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value["summary"] for key, value in combined.items()}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
