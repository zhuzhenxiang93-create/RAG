from __future__ import annotations

from legalmind.data.contracts import ProcessedCase
from legalmind.data.deduplicate import hamming_distance, normalized_fingerprint, simhash64
from legalmind.data.leakage import strip_target_leakage
from legalmind.data.normalize import normalize_text


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


def test_normalized_fingerprint_collapses_width_variants() -> None:
    assert normalized_fingerprint("金额１２３元") == normalized_fingerprint("金额123元")


def test_simhash_is_stable_and_hamming_is_symmetric() -> None:
    left = simhash64("被告人持刀造成被害人轻伤")
    right = simhash64("被告人持刀造成被害人轻伤")
    assert left == right
    assert hamming_distance(left, right) == hamming_distance(right, left) == 0
