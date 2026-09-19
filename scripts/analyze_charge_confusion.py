from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from legalmind.models.metrics import sigmoid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--label-mapping", type=Path, required=True)
    parser.add_argument("--charges", nargs="+", default=["盗窃", "抢夺", "抢劫"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--max-review", type=int, default=200)
    args = parser.parse_args()

    logits = np.load(args.results_dir / "test_logits.npy")
    labels = np.load(args.results_dir / "test_labels.npy")
    mapping = json.loads(args.label_mapping.read_text(encoding="utf-8"))
    charge_ids = {charge: int(mapping[charge]) for charge in args.charges}
    id_to_label = {int(value): key for key, value in mapping.items()}
    rows = [json.loads(line) for line in args.dataset.open(encoding="utf-8") if line.strip()]
    if not (len(rows) == len(logits) == len(labels)):
        raise ValueError("dataset, logits and labels are not aligned")

    probabilities = sigmoid(logits)
    columns = [*args.charges, "其他"]
    matrix = {truth: {prediction: 0 for prediction in columns} for truth in args.charges}
    errors = []
    evaluated = 0
    for index, row in enumerate(rows):
        true_charges = [charge for charge, label_id in charge_ids.items() if labels[index, label_id] > 0.5]
        if len(true_charges) != 1:
            continue
        truth = true_charges[0]
        prediction_id = int(np.argmax(probabilities[index]))
        prediction = id_to_label.get(prediction_id, "其他")
        bucket = prediction if prediction in charge_ids else "其他"
        matrix[truth][bucket] += 1
        evaluated += 1
        if prediction != truth:
            errors.append(
                {
                    "record_id": row.get("case_id", f"row-{index}"),
                    "true_charge": truth,
                    "predicted_charge": prediction,
                    "true_probability": float(probabilities[index, charge_ids[truth]]),
                    "predicted_probability": float(probabilities[index, prediction_id]),
                    "fact": row.get("fact", ""),
                }
            )
    errors.sort(key=lambda item: item["predicted_probability"], reverse=True)
    summary = {
        "charges": args.charges,
        "evaluated_single_target_samples": evaluated,
        "confusion_matrix": matrix,
        "review_queue_size": min(len(errors), args.max_review),
        "definition": "Rows have exactly one true label among the three target charges; prediction is global argmax.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.review_queue.parent.mkdir(parents=True, exist_ok=True)
    with args.review_queue.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(errors[0]) if errors else ["record_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(errors[: args.max_review])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
