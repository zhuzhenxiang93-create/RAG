from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.clip(logits, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def binarize(
    probabilities: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
    ensure_one: bool = True,
) -> np.ndarray:
    predictions = (probabilities >= thresholds).astype(np.int32)
    if ensure_one:
        empty_rows = np.where(predictions.sum(axis=1) == 0)[0]
        predictions[empty_rows, probabilities[empty_rows].argmax(axis=1)] = 1
    return predictions


def multilabel_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
) -> dict[str, float]:
    from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

    predictions = binarize(sigmoid(logits), thresholds)
    labels = labels.astype(np.int32)
    support = labels.sum(axis=0)
    per_label_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    supported = support > 0
    return {
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1_supported": float(per_label_f1[supported].mean()) if supported.any() else 0.0,
        "micro_precision": float(
            precision_score(labels, predictions, average="micro", zero_division=0)
        ),
        "micro_recall": float(recall_score(labels, predictions, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(labels, predictions)),
        "labels_supported": int(supported.sum()),
        "labels_total": int(labels.shape[1]),
    }


def multilabel_report(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
    label_names: dict[int, str] | None = None,
) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score

    labels = labels.astype(np.int32)
    predictions = binarize(sigmoid(logits), thresholds)
    precision = precision_score(labels, predictions, average=None, zero_division=0)
    recall = recall_score(labels, predictions, average=None, zero_division=0)
    f1 = f1_score(labels, predictions, average=None, zero_division=0)
    support = labels.sum(axis=0)
    predicted = predictions.sum(axis=0)
    per_label = [
        {
            "label_id": label_id,
            "label": (label_names or {}).get(label_id, str(label_id)),
            "support": int(support[label_id]),
            "predicted_positives": int(predicted[label_id]),
            "precision": float(precision[label_id]),
            "recall": float(recall[label_id]),
            "f1": float(f1[label_id]),
        }
        for label_id in range(labels.shape[1])
    ]

    groups = {}
    for name, mask in {
        "unsupported": support == 0,
        "rare_1_5": (support >= 1) & (support <= 5),
        "medium_6_20": (support >= 6) & (support <= 20),
        "frequent_21_plus": support >= 21,
    }.items():
        groups[name] = {
            "labels": int(mask.sum()),
            "macro_f1": float(f1[mask].mean()) if mask.any() else None,
        }
    return {
        "summary": multilabel_metrics(logits, labels, thresholds),
        "support_groups": groups,
        "per_label": per_label,
    }


def tune_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray | None = None,
    min_positive_support: int = 5,
    min_negative_support: int = 5,
) -> np.ndarray:
    from sklearn.metrics import f1_score

    candidates = candidates if candidates is not None else np.arange(0.1, 0.91, 0.05)
    thresholds = np.full(probabilities.shape[1], 0.5, dtype=np.float32)
    for label_id in range(probabilities.shape[1]):
        target = labels[:, label_id]
        positives = int(target.sum())
        negatives = int(target.size - positives)
        if positives < min_positive_support or negatives < min_negative_support:
            continue
        scores = [
            f1_score(target, probabilities[:, label_id] >= threshold, zero_division=0)
            for threshold in candidates
        ]
        thresholds[label_id] = float(candidates[int(np.argmax(scores))])
    return thresholds
