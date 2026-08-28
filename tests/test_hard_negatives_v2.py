from __future__ import annotations

from legalmind.retrieval.hard_negatives import classify_negative_type


def test_same_charge_different_behavior_is_explicit() -> None:
    value = classify_negative_type(
        "被告人共同故意持刀致人重伤",
        "被告人推搡他人",
        {"故意伤害"},
        {"故意伤害"},
    )
    assert value == "same_charge_different_behavior"


def test_completed_and_attempted_are_distinguished() -> None:
    value = classify_negative_type(
        "被告人盗窃财物后被抓获",
        "被告人着手盗窃但未遂",
        {"盗窃"},
        {"抢夺"},
    )
    assert value == "completed_vs_attempted"
