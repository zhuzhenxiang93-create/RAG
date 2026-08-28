from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def micro_f1_from_label_sets(
    references: list[list[int]], predictions: list[list[int]], num_labels: int
) -> float:
    if len(references) != len(predictions):
        raise ValueError("Reference and prediction row counts differ")
    gold = np.zeros((len(references), num_labels), dtype=np.int8)
    pred = np.zeros_like(gold)
    for row, values in enumerate(references):
        gold[row, values] = 1
    for row, values in enumerate(predictions):
        valid = [value for value in values if 0 <= value < num_labels]
        pred[row, valid] = 1
    return float(f1_score(gold, pred, average="micro", zero_division=0))


def read_rows(path: Path, field: str) -> list[list[int]]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if field == "reference":
                values.append([int(value) for value in row["accusation_ids"]])
            else:
                values.append([int(value) for value in row["predicted_label_ids"]])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--label-mapping", default="data/processed_v2/label_mapping.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mapping = json.loads(Path(args.label_mapping).read_text(encoding="utf-8"))
    score = micro_f1_from_label_sets(
        read_rows(Path(args.references), "reference"),
        read_rows(Path(args.predictions), "prediction"),
        len(mapping),
    )
    result = {
        "primary_metric": "micro_f1",
        "micro_f1": score,
        "rows": len(read_rows(Path(args.references), "reference")),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
