"""Dependency-free classification metrics used by training and evaluation."""

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence


def classification_metrics(
    gold: Sequence[int],
    predicted: Sequence[int],
    confidences: Sequence[float] = (),
    *,
    label_count: int,
    ece_bins: int = 10,
) -> Dict:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    if not gold:
        raise ValueError("at least one prediction is required")
    matrix = [[0 for _ in range(label_count)] for _ in range(label_count)]
    for expected, actual in zip(gold, predicted):
        matrix[int(expected)][int(actual)] += 1
    accuracy = sum(matrix[i][i] for i in range(label_count)) / len(gold)
    per_label = {}
    f1_values: List[float] = []
    for label_id in range(label_count):
        true_positive = matrix[label_id][label_id]
        false_positive = sum(matrix[row][label_id] for row in range(label_count)) - true_positive
        false_negative = sum(matrix[label_id]) - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        support = sum(matrix[label_id])
        per_label[str(label_id)] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
        if support:
            f1_values.append(f1)
    result = {
        "sample_count": len(gold),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "per_label": per_label,
        "confusion_matrix": matrix,
    }
    if confidences:
        if len(confidences) != len(gold):
            raise ValueError("confidence length differs from predictions")
        buckets = defaultdict(list)
        for expected, actual, confidence in zip(gold, predicted, confidences):
            bucket = min(int(float(confidence) * ece_bins), ece_bins - 1)
            buckets[bucket].append((int(expected == actual), float(confidence)))
        ece = 0.0
        for values in buckets.values():
            bucket_accuracy = sum(value[0] for value in values) / len(values)
            bucket_confidence = sum(value[1] for value in values) / len(values)
            ece += len(values) / len(gold) * abs(bucket_accuracy - bucket_confidence)
        result["ece"] = round(ece, 6)
    return result


def macro_f1_from_logits(logits: Iterable[Iterable[float]], labels: Sequence[int]) -> Dict:
    predictions = [max(range(len(row)), key=lambda index: row[index]) for row in logits]
    count = max(max(labels), max(predictions)) + 1
    metrics = classification_metrics(labels, predictions, label_count=count)
    return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}

