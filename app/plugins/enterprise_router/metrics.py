"""Field-level metrics for generative multi-task routing."""

from typing import Dict, Iterable, Optional, Sequence, Set

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
    per_label_scores = {}
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = sum(label in gold and label in actual for gold, actual in zip(expected, predicted))
        fp = sum(label not in gold and label in actual for gold, actual in zip(expected, predicted))
        fn = sum(label in gold and label not in actual for gold, actual in zip(expected, predicted))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        score = _f1(tp, fp, fn)
        per_label.append(score)
        per_label_scores[label] = round(score, 6)
    return {
        "micro_f1": round(_f1(total_tp, total_fp, total_fn), 6),
        "macro_f1": round(sum(per_label) / len(per_label), 6),
        "per_label_f1": per_label_scores,
    }


def routing_metrics(
    expected: Sequence[RouteDecision],
    predicted: Sequence[Optional[RouteDecision]],
) -> Dict:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths differ")
    attempted_count = len(expected)
    if attempted_count < 1:
        raise ValueError("at least one prediction is required")
    valid_count = sum(item is not None for item in predicted)
    exact = sum(gold == actual for gold, actual in zip(expected, predicted))
    type_correct = sum(
        actual is not None and gold.question_type == actual.question_type
        for gold, actual in zip(expected, predicted)
    )
    type_f1 = {}
    for label in QUESTION_TYPES:
        tp = sum(
            actual is not None
            and gold.question_type == label
            and actual.question_type == label
            for gold, actual in zip(expected, predicted)
        )
        fp = sum(
            actual is not None
            and gold.question_type != label
            and actual.question_type == label
            for gold, actual in zip(expected, predicted)
        )
        fn = sum(
            gold.question_type == label
            and (actual is None or actual.question_type != label)
            for gold, actual in zip(expected, predicted)
        )
        type_f1[label] = round(_f1(tp, fp, fn), 6)
    source_scores = multilabel_f1(
        [set(item.sources) for item in expected],
        [set(item.sources) if item is not None else set() for item in predicted],
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
            actual is not None
            and bool(getattr(gold, field))
            and bool(getattr(actual, field))
            for gold, actual in zip(expected, predicted)
        )
        fp = sum(
            actual is not None
            and not bool(getattr(gold, field))
            and bool(getattr(actual, field))
            for gold, actual in zip(expected, predicted)
        )
        fn = sum(
            bool(getattr(gold, field))
            and (actual is None or not bool(getattr(actual, field)))
            for gold, actual in zip(expected, predicted)
        )
        boolean_f1[field] = round(_f1(tp, fp, fn), 6)
    return {
        "sample_count": attempted_count,
        "valid_prediction_count": valid_count,
        "json_valid_rate": round(valid_count / attempted_count, 6),
        "exact_route_accuracy": round(exact / attempted_count, 6),
        "question_type_accuracy": round(type_correct / attempted_count, 6),
        "question_type_macro_f1": round(
            sum(type_f1.values()) / len(type_f1), 6
        ),
        "question_type_per_class_f1": type_f1,
        "source_micro_f1": source_scores["micro_f1"],
        "source_macro_f1": source_scores["macro_f1"],
        "source_per_class_f1": source_scores["per_label_f1"],
        "boolean_f1": boolean_f1,
        "supported_question_types": list(QUESTION_TYPES),
    }
