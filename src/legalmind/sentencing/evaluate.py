from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_recall_fscore_support,
)

from legalmind.sentencing.labels import labels_from_row, sentence_type_from_labels
from legalmind.sentencing.model import SentencingBaseline, model_text


def _regression_metrics(truth: list[int], predicted: list[int]) -> dict:
    if not truth:
        return {"rows": 0, "mae": None, "median_ae": None}
    return {
        "rows": len(truth),
        "mae": float(mean_absolute_error(truth, predicted)),
        "median_ae": float(median_absolute_error(truth, predicted)),
    }


def evaluate(model: SentencingBaseline, rows: list[dict]) -> dict:
    type_truth, type_predicted = [], []
    month_truth, month_predicted, month_in_range = [], [], []
    fine_binary_truth, fine_binary_predicted = [], []
    fine_truth, fine_predicted, fine_in_range = [], [], []
    grouped: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"month_truth": [], "month_predicted": [], "fine_truth": [], "fine_predicted": []}
    )
    missing = Counter()
    for row in rows:
        labels = labels_from_row(row)
        accusations = labels.get("accusations", [])
        prediction = model.predict(row["fact"], accusations)
        true_type = sentence_type_from_labels(labels).value
        if true_type != "unknown":
            type_truth.append(true_type)
            type_predicted.append(prediction["sentence_type"])
        else:
            missing["sentence_type"] += 1

        months = labels.get("imprisonment_months")
        if true_type == "fixed_term" and isinstance(months, int):
            matrix = model.vectorizer.transform([model_text(row["fact"], accusations)])
            predicted_months, (lower, upper) = model.predict_imprisonment_given_fixed(matrix)
            month_truth.append(months)
            month_predicted.append(predicted_months)
            month_in_range.append(lower <= months <= upper)
            if accusations:
                grouped[accusations[0]]["month_truth"].append(months)
                grouped[accusations[0]]["month_predicted"].append(predicted_months)

        fine = labels.get("fine")
        fine_status = fine.get("status") if isinstance(fine, dict) else None
        if isinstance(fine, dict):
            fine = fine.get("amount") if fine_status in {"zero", "positive"} else None
        if isinstance(fine, int) and fine >= 0:
            fine_binary_truth.append(int(fine > 0))
            fine_binary_predicted.append(int(bool(prediction["fine_imposed"])))
            if fine > 0:
                matrix = model.vectorizer.transform([model_text(row["fact"], accusations)])
                predicted_fine, (lower, upper) = model.predict_fine_amount_given_imposed(matrix)
                fine_truth.append(fine)
                fine_predicted.append(predicted_fine)
                fine_in_range.append(lower <= fine <= upper)
                if accusations:
                    grouped[accusations[0]]["fine_truth"].append(fine)
                    grouped[accusations[0]]["fine_predicted"].append(predicted_fine)
        else:
            missing["fine"] += 1

    type_labels = sorted(set(type_truth) | set(type_predicted))
    precision, recall, fine_f1, _ = precision_recall_fscore_support(
        fine_binary_truth, fine_binary_predicted, average="binary", zero_division=0
    )
    most_common = Counter(
        labels_from_row(row).get("accusations", ["unknown"])[0]
        for row in rows
        if labels_from_row(row).get("accusations")
    ).most_common(20)
    grouped_metrics = {}
    for accusation, _ in most_common:
        values = grouped[accusation]
        grouped_metrics[accusation] = {
            "imprisonment": _regression_metrics(values["month_truth"], values["month_predicted"]),
            "fine_positive": _regression_metrics(values["fine_truth"], values["fine_predicted"]),
        }
    return {
        "test_rows": len(rows),
        "accusation_prediction": {
            "status": "evaluated_separately_by_existing_charge_classifier",
            "note": "This sentencing baseline conditions on optional accusations and does not replace charge evaluation.",
        },
        "sentence_type": {
            "rows": len(type_truth),
            "accuracy": float(accuracy_score(type_truth, type_predicted)),
            "macro_f1": float(f1_score(type_truth, type_predicted, average="macro")),
            "labels": type_labels,
            "confusion_matrix": confusion_matrix(
                type_truth, type_predicted, labels=type_labels
            ).tolist(),
        },
        "imprisonment": {
            **_regression_metrics(month_truth, month_predicted),
            "bucket_interval_accuracy": float(np.mean(month_in_range)) if month_in_range else None,
        },
        "fine_imposed": {
            "rows": len(fine_binary_truth),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(fine_f1),
        },
        "fine_positive_amount": {
            **_regression_metrics(fine_truth, fine_predicted),
            "bucket_interval_accuracy": float(np.mean(fine_in_range)) if fine_in_range else None,
        },
        "missing_labels": dict(missing),
        "major_accusation_groups": grouped_metrics,
    }


def write_evaluation(report: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
