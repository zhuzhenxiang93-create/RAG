from legalmind.data.sampling import select_multilabel_subset


def test_coverage_first_subset_keeps_rare_labels() -> None:
    records = [
        {"accusation_ids": [0]},
        {"accusation_ids": [0]},
        {"accusation_ids": [0, 1]},
        {"accusation_ids": [2]},
        {"accusation_ids": [3]},
        {"accusation_ids": [0]},
    ]

    sampled, report = select_multilabel_subset(records, 4, 4, seed=7)

    assert len(sampled) == 4
    assert report["labels_present"] == 4
    assert report["labels_missing"] == []
    assert report["label_support"][1] == 1
    assert report["label_support"][2] == 1
    assert report["label_support"][3] == 1


def test_subset_selection_is_reproducible() -> None:
    records = [{"accusation_ids": [index % 3]} for index in range(30)]

    _, first = select_multilabel_subset(records, 12, 3, seed=42)
    _, second = select_multilabel_subset(records, 12, 3, seed=42)

    assert first["selected_indices_sha256"] == second["selected_indices_sha256"]


def test_subset_size_is_exact() -> None:
    records = [{"accusation_ids": [index % 7, (index * 3) % 11]} for index in range(250)]

    sampled, report = select_multilabel_subset(records, 100, 11, seed=42)

    assert len(sampled) == 100
    assert report["rows"] == 100
