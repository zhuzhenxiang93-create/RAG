from __future__ import annotations

import argparse
import json
from pathlib import Path

from legalmind.evaluation.evaluate_classification import micro_f1_from_label_sets
from legalmind.generation.structured import (
    extract_json_object,
    parse_and_validate_json,
    parse_simplified_json,
    validate_citations,
)


def evaluate(path: Path, label_mapping: dict[str, int]) -> dict:
    rows = json_parsed = schema_valid = simplified_valid = repaired_valid = 0
    references: list[list[int]] = []
    predictions: list[list[int]] = []
    grounded = unsupported = 0
    simplified_predictions: list[list[int]] = []
    repaired_predictions: list[list[int]] = []
    error_counts = {
        "no_or_truncated_json": 0,
        "invalid_json": 0,
        "full_schema_invalid": 0,
        "simplified_schema_invalid": 0,
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            references.append(
                [
                    label_mapping[label]
                    for label in row["reference_accusations"]
                    if label in label_mapping
                ]
            )
            try:
                json.loads(extract_json_object(row["output"]))
                json_parsed += 1
            except ValueError:
                error_counts["no_or_truncated_json"] += 1
            except json.JSONDecodeError:
                error_counts["invalid_json"] += 1
            try:
                value = parse_and_validate_json(row["output"])
                schema_valid += 1
                predictions.append(
                    [
                        label_mapping[label]
                        for label in value.predicted_accusations
                        if label in label_mapping
                    ]
                )
                citation = validate_citations(
                    value,
                    set(row.get("retrieved_case_ids", [])),
                    set(row.get("retrieved_articles", [])),
                )
                grounded += int(bool(citation["valid"]))
                unsupported += len(citation["invalid_case_ids"]) + len(citation["invalid_articles"])
            except (ValueError, TypeError, json.JSONDecodeError):
                predictions.append([])
                error_counts["full_schema_invalid"] += 1
            try:
                value = parse_simplified_json(row["output"])
                simplified_valid += 1
                simplified_predictions.append(
                    [
                        label_mapping[label]
                        for label in value.candidate_accusations
                        if label in label_mapping
                    ]
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                simplified_predictions.append([])
                error_counts["simplified_schema_invalid"] += 1
            try:
                value = parse_simplified_json(row["output"], repair=True)
                repaired_valid += 1
                repaired_predictions.append(
                    [
                        label_mapping[label]
                        for label in value.candidate_accusations
                        if label in label_mapping
                    ]
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                repaired_predictions.append([])
    return {
        "rows": rows,
        "json_parse_rate": json_parsed / rows if rows else 0.0,
        "schema_pass_rate": schema_valid / rows if rows else 0.0,
        "accusation_micro_f1": micro_f1_from_label_sets(references, predictions, len(label_mapping))
        if rows
        else 0.0,
        "citation_grounded_rate": grounded / rows if rows else 0.0,
        "unsupported_citations": unsupported,
        "simplified_schema_pass_rate": simplified_valid / rows if rows else 0.0,
        "simplified_accusation_micro_f1": micro_f1_from_label_sets(
            references, simplified_predictions, len(label_mapping)
        )
        if rows
        else 0.0,
        "canonical_repair_schema_pass_rate": repaired_valid / rows if rows else 0.0,
        "canonical_repair_accusation_micro_f1": micro_f1_from_label_sets(
            references, repaired_predictions, len(label_mapping)
        )
        if rows
        else 0.0,
        "error_counts": error_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--label-mapping", default="data/processed_v2/label_mapping.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mapping = json.loads(Path(args.label_mapping).read_text(encoding="utf-8"))
    metrics = evaluate(Path(args.predictions), mapping)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
