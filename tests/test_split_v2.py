from __future__ import annotations

from legalmind.data.split import assert_group_isolation, multilabel_group_split


def rows(count: int = 100) -> list[dict]:
    return [
        {
            "case_id": f"c{index}",
            "accusation_ids": [index % 5],
            "cleaning_metadata": {"dedup_group_id": f"g{index}"},
        }
        for index in range(count)
    ]


def test_group_split_is_reproducible_and_isolated() -> None:
    first = multilabel_group_split(rows(), 5, 0.1, 0.1, 42)
    second = multilabel_group_split(rows(), 5, 0.1, 0.1, 42)
    assert {key: [row["case_id"] for row in value] for key, value in first.items()} == {
        key: [row["case_id"] for row in value] for key, value in second.items()
    }
    assert_group_isolation(first)


def test_group_isolation_detects_leakage() -> None:
    values = {
        "train": [{"cleaning_metadata": {"dedup_group_id": "same"}}],
        "validation": [{"cleaning_metadata": {"dedup_group_id": "same"}}],
    }
    try:
        assert_group_isolation(values)
    except ValueError:
        return
    raise AssertionError("group leakage must fail validation")
