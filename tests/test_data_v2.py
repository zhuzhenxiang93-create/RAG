from __future__ import annotations

from legalmind.data.build_dataset import penalty_labels, representative_penalty
from legalmind.data.contracts import ProcessedCase
from legalmind.data.deduplicate import hamming_distance, normalized_fingerprint, simhash64
from legalmind.data.leakage import strip_target_leakage
from legalmind.data.normalize import normalize_text
from legalmind.data.statistics import split_statistics


def test_nfkc_and_whitespace_normalization() -> None:
    assert normalize_text("ＡＢＣ\u3000  １２３") == "ABC 123"


def test_target_leakage_is_masked() -> None:
    result = strip_target_leakage("公诉机关认为其行为构成盗窃罪，提请依法惩处。")
    assert result.removed
    assert "盗窃罪" not in result.text
    assert "[罪名结论已脱敏]" in result.text


def test_processed_contract_rejects_empty_fact() -> None:
    try:
        ProcessedCase.model_validate(
            {
                "case_id": "x",
                "source_dataset": "test",
                "source_split": "train",
                "fact": " ",
                "labels": {},
                "cleaning_metadata": {
                    "dedup_group_id": "g",
                    "fact_sha256": "a" * 64,
                },
                "length_metadata": {"characters": 0},
            }
        )
    except ValueError:
        return
    raise AssertionError("empty facts must be rejected")


def test_fine_is_preserved_in_processed_contract() -> None:
    case = ProcessedCase.model_validate(
        {
            "case_id": "x",
            "source_dataset": "test",
            "source_split": "train",
            "fact": "被告人盗窃手机一部。",
            "labels": {
                "accusations": ["盗窃"],
                "imprisonment_months": 8,
                "fine": 1000,
            },
            "cleaning_metadata": {
                "dedup_group_id": "g",
                "fact_sha256": "a" * 64,
            },
            "length_metadata": {"characters": 11},
        }
    )
    assert case.labels.fine == 1000


def test_penalty_labels_preserves_numeric_fine_and_rejects_invalid_value() -> None:
    assert penalty_labels({"fine": "1000"})["fine"] == 1000
    assert penalty_labels({"fine": "unknown"})["fine"] is None
    assert penalty_labels({"fine": -1000})["fine"] is None


def test_split_statistics_audits_fine_coverage_and_distribution() -> None:
    records = [
        {
            "accusation_ids": [0],
            "labels": {"fine": 0},
            "cleaning_metadata": {"target_leakage_removed": False},
            "length_metadata": {"characters": 10},
        },
        {
            "accusation_ids": [0],
            "labels": {"fine": 1000},
            "cleaning_metadata": {"target_leakage_removed": True},
            "length_metadata": {"characters": 20},
        },
        {
            "accusation_ids": [1],
            "labels": {"fine": None},
            "cleaning_metadata": {"target_leakage_removed": False},
            "length_metadata": {"characters": 30},
        },
    ]
    fine = split_statistics(records, num_labels=2)["fine"]
    assert fine["present_rows"] == 2
    assert fine["missing_rows"] == 1
    assert fine["zero_rows"] == 1
    assert fine["max"] == 1000


def test_duplicate_penalty_conflicts_are_detected_and_modal_value_selected() -> None:
    selected, conflict = representative_penalty(
        [
            {"imprisonment": 8, "fine": 1000},
            {"imprisonment": 8, "fine": 1000},
            {"imprisonment": 10, "fine": 2000},
        ]
    )
    assert conflict is True
    assert selected["imprisonment_months"] == 8
    assert selected["fine"] == 1000


def test_sentencing_conclusions_are_removed_but_crime_amount_is_preserved() -> None:
    text = "被告人盗窃人民币5000元，已经退赃。判处有期徒刑八个月，并处罚金一千元。缓刑一年。"
    result = strip_target_leakage(text)
    assert "盗窃人民币5000元" in result.text
    assert "有期徒刑八个月" not in result.text
    assert "罚金一千元" not in result.text
    assert "缓刑一年" not in result.text
    assert "explicit_term_sentence" in result.leakage_types


def test_partially_anonymized_sentencing_recommendation_is_removed() -> None:
    result = strip_target_leakage("公诉机关建议判处被告人××至四年，并处罚金。被告人已经退赃。")
    assert "判处被告人××至四年" not in result.text
    assert "并处罚金" not in result.text
    assert "被告人已经退赃" in result.text
    assert "generic_sentence_after_verdict_verb" in result.leakage_types
    assert "explicit_fine_sentence" in result.leakage_types


def test_normalized_fingerprint_collapses_width_variants() -> None:
    assert normalized_fingerprint("金额１２３元") == normalized_fingerprint("金额123元")


def test_simhash_is_stable_and_hamming_is_symmetric() -> None:
    left = simhash64("被告人持刀造成被害人轻伤")
    right = simhash64("被告人持刀造成被害人轻伤")
    assert left == right
    assert hamming_distance(left, right) == hamming_distance(right, left) == 0
