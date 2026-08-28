from legalmind.evaluation.evaluate_classification import micro_f1_from_label_sets


def test_micro_f1_exact_match() -> None:
    assert micro_f1_from_label_sets([[0], [1, 2]], [[0], [1, 2]], 3) == 1.0
