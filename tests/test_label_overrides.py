import json

from legalmind.data.label_overrides import apply_label_overrides, load_label_overrides
from legalmind.pipeline.core import select_retrieval_label_scores
from legalmind.schemas import CaseChunk, LabelScore


def _chunk(accusations: list[str]) -> CaseChunk:
    return CaseChunk(
        chunk_id="case-1:c0",
        case_id="case-1",
        text="人工构造的脱敏案情。",
        chunk_index=0,
        start_char=0,
        end_char=10,
        accusations=accusations,
    )


def _write_override(tmp_path, status: str):
    path = tmp_path / "overrides.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "original_accusations": ["抢夺"],
                "corrected_accusations": ["盗窃"],
                "status": status,
            },
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def test_pending_override_is_not_applied(tmp_path) -> None:
    rows, audit = apply_label_overrides(
        [_chunk(["抢夺"])], load_label_overrides(_write_override(tmp_path, "pending"))
    )
    assert rows[0].accusations == ["抢夺"]
    assert audit["pending"] == 1
    assert audit["applied_chunks"] == 0


def test_approved_override_is_applied_without_mutating_source(tmp_path) -> None:
    source = _chunk(["抢夺"])
    rows, audit = apply_label_overrides(
        [source], load_label_overrides(_write_override(tmp_path, "approved"))
    )
    assert source.accusations == ["抢夺"]
    assert rows[0].accusations == ["盗窃"]
    assert audit["applied_chunks"] == 1


def test_uncertain_prediction_keeps_three_retrieval_candidates() -> None:
    scores = [
        LabelScore(label_id=index, label=f"罪名{index}", probability=0.5 - index * 0.1)
        for index in range(5)
    ]
    selected = select_retrieval_label_scores(scores, used_fallback=True)
    assert [item.label_id for item in selected] == [0, 1, 2]
    assert select_retrieval_label_scores(scores, used_fallback=False) == scores
