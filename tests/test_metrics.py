import numpy as np

from legalmind.models.metrics import multilabel_metrics, tune_thresholds


def test_macro_f1_supported_excludes_absent_labels() -> None:
    labels = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int32)
    logits = np.array([[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0]])

    metrics = multilabel_metrics(logits, labels)

    assert metrics["macro_f1"] == 2 / 3
    assert metrics["macro_f1_supported"] == 1.0
    assert metrics["labels_supported"] == 2
    assert metrics["labels_total"] == 3


def test_threshold_tuning_keeps_default_for_tiny_support() -> None:
    probabilities = np.array([[0.4], [0.3], [0.2], [0.1], [0.05], [0.01]])
    labels = np.array([[1], [0], [0], [0], [0], [0]])

    thresholds = tune_thresholds(probabilities, labels)

    assert thresholds[0] == 0.5
