from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

from legalmind.sentencing.labels import sentence_type_from_labels, targets_from_labels
from legalmind.sentencing.schemas import PredictedAccusation, SentenceType, SentencingRequest
from legalmind.sentencing.service import SentencingService, extract_factors


class FakeModel:
    metadata: ClassVar[dict] = {"model_type": "fake"}

    def predict(self, fact: str, accusations: list[str] | None = None) -> dict:
        return {
            "sentence_type": "fixed_term",
            "sentence_type_probability": 0.8,
            "imprisonment_months": 8,
            "imprisonment_range": (6, 11),
            "fine_imposed": True,
            "fine_probability": 0.75,
            "fine_amount": 1000,
            "fine_range": (1000, 1999),
        }


class FakeChargePredictor:
    def predict(self, fact: str):
        return SimpleNamespace(labels=[SimpleNamespace(label="盗窃", probability=0.91)])


def test_hierarchical_target_construction() -> None:
    assert sentence_type_from_labels({"death_penalty": True}) is SentenceType.death
    assert sentence_type_from_labels({"life_imprisonment": True}) is SentenceType.life
    targets = targets_from_labels({"imprisonment_months": 8, "fine": 1000})
    assert targets.sentence_type == "fixed_term"
    assert targets.fine_imposed is True
    assert targets.fine_bucket is not None


def test_only_accusation_returns_insufficient_information() -> None:
    response = SentencingService(FakeModel()).predict(
        SentencingRequest(fact="", accusation=["盗窃"])
    )
    assert response.status == "insufficient_information"
    assert response.sentence.imprisonment_months is None
    assert response.sentence.fine.amount is None
    assert response.requires_manual_review is True


def test_structured_prediction_and_charge_adapter() -> None:
    service = SentencingService(FakeModel(), charge_predictor=FakeChargePredictor())
    response = service.predict(
        {
            "fact": "当事人盗窃手机一部，价值人民币5000元，主动投案并退赃，取得谅解。",
            "top_k": 0,
        }
    )
    assert response.predicted_accusations[0].name == "盗窃"
    assert response.sentence.imprisonment_months == 8
    assert response.sentence.fine.amount == 1000
    assert response.sentence.probation.predicted is None
    assert "自首" in response.key_sentencing_factors
    assert "退赃退赔" in response.key_sentencing_factors


def test_consistency_removes_amount_when_fine_not_imposed() -> None:
    model = FakeModel()

    def predict(fact: str, accusations=None):
        value = FakeModel.predict(model, fact, accusations)
        value.update({"fine_imposed": False, "fine_probability": 0.2})
        return value

    model.predict = predict
    response = SentencingService(model).predict(
        {"fact": "当事人故意损坏他人财物，案件事实材料较为完整，已经赔偿损失。"}
    )
    assert response.sentence.fine.imposed is False
    assert response.sentence.fine.amount is None
    assert response.sentence.fine.amount_range is None


def test_factor_extraction_does_not_treat_sentence_as_input_requirement() -> None:
    factors = extract_factors("涉案价值人民币3万元，案发后如实供述并认罪认罚。")
    assert factors == ["坦白", "认罪认罚", "涉案金额"]


def test_upstream_classifier_probability_is_preserved() -> None:
    response = SentencingService(FakeModel()).predict(
        SentencingRequest(
            fact="当事人秘密窃取手机一部，价值人民币5000元，案情事实材料完整。",
            predicted_accusations=[PredictedAccusation(name="盗窃", probability=0.31)],
            top_k=0,
        )
    )
    assert response.predicted_accusations[0].probability == 0.31
