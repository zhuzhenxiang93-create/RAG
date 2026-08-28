from __future__ import annotations

import numpy as np


def multilabel_group_split(
    records: list[dict],
    num_labels: int,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    """Stratify one canonical row per dedup group and assert group isolation."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    groups = [row["cleaning_metadata"]["dedup_group_id"] for row in records]
    if len(groups) != len(set(groups)):
        raise ValueError("Input must contain one merged row per dedup_group_id")
    labels = np.zeros((len(records), num_labels), dtype=np.int8)
    for row_index, row in enumerate(records):
        labels[row_index, row["accusation_ids"]] = 1
    indices = np.arange(len(records))
    outer = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=validation_ratio + test_ratio, random_state=seed
    )
    train_indices, held_indices = next(outer.split(indices, labels))
    held_labels = labels[held_indices]
    inner = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio / (validation_ratio + test_ratio),
        random_state=seed + 1,
    )
    validation_local, test_local = next(inner.split(held_indices, held_labels))
    splits = {
        "train": [records[int(index)] for index in train_indices],
        "validation": [records[int(held_indices[index])] for index in validation_local],
        "test": [records[int(held_indices[index])] for index in test_local],
    }
    assert_group_isolation(splits)
    for split, rows in splits.items():
        for row in rows:
            row["source_split"] = split
    return splits


def assert_group_isolation(splits: dict[str, list[dict]]) -> None:
    groups = {
        split: {row["cleaning_metadata"]["dedup_group_id"] for row in rows}
        for split, rows in splits.items()
    }
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"dedup group leakage between {left}/{right}: {len(overlap)}")
