"""Retrieval and selective-answering metrics."""

import math
from typing import Dict, Iterable, List


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    relevant_set = set(relevant)
    return sum(item in relevant_set for item in retrieved[:k]) / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, 1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], 1)
        if item in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: List[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def binary_precision_recall(
    expected_positive: List[bool], predicted_positive: List[bool]
) -> Dict[str, float]:
    true_positive = sum(expected and predicted for expected, predicted in zip(
        expected_positive, predicted_positive
    ))
    false_positive = sum(not expected and predicted for expected, predicted in zip(
        expected_positive, predicted_positive
    ))
    false_negative = sum(expected and not predicted for expected, predicted in zip(
        expected_positive, predicted_positive
    ))
    precision = true_positive / (true_positive + false_positive) if (
        true_positive + false_positive
    ) else 0.0
    recall = true_positive / (true_positive + false_negative) if (
        true_positive + false_negative
    ) else 0.0
    return {"precision": precision, "recall": recall}
