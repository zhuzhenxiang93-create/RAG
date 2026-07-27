"""Field-level metrics for generative multi-task routing."""

from typing import Dict, Iterable, List, Sequence, Set

from app.plugins.enterprise_router.schema import QUESTION_TYPES, SOURCE_TYPES, RouteDecision


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def multilabel_f1(
    expected: Sequence[Set[str]], predicted: Sequence[Set[str]], labels: Iterable[str]
) -> Dict[str, float]:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    per_label = []
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = sum(label in gold and label in actual for gold, actual in zip(expected, predicted))
        fp = sum(label not in gold and label in actual for gold, actual in zip(expected, predicted))
        fn = sum(label in gold and label not in actual for gold, actual in zip(expected, predicted))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_label.append(_f1(tp, fp, fn))
    return {
        "micro_f1": round(_f1(total_tp, total_fp, total_fn), 6),
        "macro_f1": round(sum(per_label) / len(per_label), 6),
    }


def routing_metrics(
    expected: Sequence[RouteDecision],
    predicted: Sequence[RouteDecision],
    *,
    attempted_count: int,
) -> Dict:
    if len(expected) != len(predicted):
        raise ValueError("valid expected and predicted lengths differ")
    if attempted_count < len(expected) or attempted_count < 1:
        raise ValueError("attempted_count must include all valid predictions")
    valid_count = len(predicted)
    exact = sum(gold == actual for gold, actual in zip(expected, predicted))
    type_correct = sum(
        gold.question_type == actual.question_type
        for gold, actual in zip(expected, predicted)
    )
    source_scores = multilabel_f1(
        [set(item.sources) for item in expected],
        [set(item.sources) for item in predicted],
        SOURCE_TYPES,
    )
    boolean_fields = (
        "multi_document",
        "conflict_check",
        "completeness_required",
    )
    boolean_f1 = {}
    for field in boolean_fields:
        tp = sum(
            bool(getattr(gold, field)) and bool(getattr(actual, field))
            for gold, actual in zip(expected, predicted)
        )
        fp = sum(
            not bool(getattr(gold, field)) and bool(getattr(actual, field))
            for gold, actual in zip(expected, predicted)
        )
        fn = sum(
            bool(getattr(gold, field)) and not bool(getattr(actual, field))
            for gold, actual in zip(expected, predicted)
        )
        boolean_f1[field] = round(_f1(tp, fp, fn), 6)
    return {
        "sample_count": attempted_count,
        "valid_prediction_count": valid_count,
        "json_valid_rate": round(valid_count / attempted_count, 6),
        "exact_route_accuracy": round(exact / attempted_count, 6),
        "question_type_accuracy": round(type_correct / attempted_count, 6),
        "source_micro_f1": source_scores["micro_f1"],
        "source_macro_f1": source_scores["macro_f1"],
        "boolean_f1": boolean_f1,
        "supported_question_types": list(QUESTION_TYPES),
    }

