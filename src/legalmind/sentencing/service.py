from __future__ import annotations

import re
from typing import Any

from legalmind.sentencing.evidence import SentencingEvidenceIndex
from legalmind.sentencing.model import SentencingBaseline
from legalmind.sentencing.schemas import (
    BinaryPrediction,
    FinePrediction,
    MonthsRange,
    NumericRange,
    PredictedAccusation,
    SentencePrediction,
    SentenceType,
    SentencingRequest,
    SentencingResponse,
)

_FACTOR_PATTERNS = (
    ("自首", r"自首|主动投案"),
    ("坦白", r"坦白|如实供述"),
    ("退赃退赔", r"退赃|退赔|返还"),
    ("取得谅解", r"谅解"),
    ("累犯或前科", r"累犯|前科|曾因.*(?:判刑|刑事处罚)"),
    ("未遂", r"未遂"),
    ("认罪认罚", r"认罪认罚"),
    ("涉案金额", r"(?:价值|金额|人民币)[^，。；]{0,16}(?:元|万元)"),
)

_MISSING_INFORMATION = [
    "涉案金额或损害后果",
    "犯罪次数",
    "自首或坦白情节",
    "退赃退赔情况",
    "是否取得谅解",
    "是否属于累犯",
]


def extract_factors(fact: str) -> list[str]:
    return [label for label, pattern in _FACTOR_PATTERNS if re.search(pattern, fact)]


class SentencingService:
    def __init__(
        self,
        model: SentencingBaseline,
        evidence_index: SentencingEvidenceIndex | None = None,
        charge_predictor: Any | None = None,
        manual_review_threshold: float = 0.60,
    ):
        self.model = model
        self.evidence_index = evidence_index
        self.charge_predictor = charge_predictor
        self.manual_review_threshold = manual_review_threshold

    def _accusations(
        self, request: SentencingRequest
    ) -> tuple[list[PredictedAccusation], list[str]]:
        if request.accusation:
            accusations = [
                PredictedAccusation(name=name, probability=1.0) for name in request.accusation
            ]
            return accusations, []
        if request.predicted_accusations:
            return request.predicted_accusations, []
        if self.charge_predictor is None:
            return [], ["未配置罪名分类器；量刑模型按罪名未知模式运行。"]
        result = self.charge_predictor.predict(request.fact)
        return [
            PredictedAccusation(name=item.label, probability=item.probability)
            for item in result.labels
        ], []

    def predict(self, value: SentencingRequest | dict) -> SentencingResponse:
        request = (
            value
            if isinstance(value, SentencingRequest)
            else SentencingRequest.model_validate(value)
        )
        fact = request.fact.strip()
        if len(fact) < 20:
            return SentencingResponse(
                status="insufficient_information",
                predicted_accusations=[
                    PredictedAccusation(name=name, probability=1.0) for name in request.accusation
                ]
                or request.predicted_accusations,
                missing_information=_MISSING_INFORMATION,
                warnings=["案件事实过短，系统不会仅凭罪名编造具体刑期或罚金。"],
            )

        predicted_accusations, warnings = self._accusations(request)
        names = [item.name for item in predicted_accusations]
        raw = self.model.predict(fact, names)
        sentence_type = SentenceType(raw["sentence_type"])
        non_month_sentence = sentence_type in {
            SentenceType.death,
            SentenceType.life,
            SentenceType.exempt,
        }
        imprisonment_range = (
            MonthsRange(
                min_months=raw["imprisonment_range"][0],
                max_months=raw["imprisonment_range"][1],
            )
            if raw["imprisonment_range"]
            else None
        )
        fine_range = (
            NumericRange(min=raw["fine_range"][0], max=raw["fine_range"][1])
            if raw["fine_range"]
            else None
        )
        sentence = SentencePrediction(
            sentence_type=sentence_type,
            sentence_type_probability=raw["sentence_type_probability"],
            imprisonment_months=None if non_month_sentence else raw["imprisonment_months"],
            imprisonment_range=None if non_month_sentence else imprisonment_range,
            probation=BinaryPrediction(predicted=None, probability=None),
            fine=FinePrediction(
                imposed=raw["fine_imposed"],
                probability=raw["fine_probability"],
                amount=None if raw["fine_imposed"] is False else raw["fine_amount"],
                amount_range=None if raw["fine_imposed"] is False else fine_range,
            ),
            death_penalty=BinaryPrediction(
                predicted=sentence_type is SentenceType.death,
                probability=raw.get("death_probability", 0.0),
            ),
            life_imprisonment=BinaryPrediction(
                predicted=sentence_type is SentenceType.life,
                probability=raw.get("life_probability", 0.0),
            ),
        )
        similar_cases = (
            self.evidence_index.search(fact, names, min(request.top_k, 3))
            if self.evidence_index
            else []
        )
        confidence_parts = [raw["sentence_type_probability"]]
        if raw["fine_probability"] is not None:
            confidence_parts.append(max(raw["fine_probability"], 1.0 - raw["fine_probability"]))
        if similar_cases:
            confidence_parts.append(min(1.0, similar_cases[0].similarity_score))
        confidence = float(sum(confidence_parts) / len(confidence_parts))
        if not self.evidence_index:
            warnings.append("未加载量刑类案索引，本次结果没有相似案件校准。")
        warnings.append("现有 CAIL 标签无法可靠区分有期徒刑、拘役和管制，也不支持缓刑预测。")
        requires_review = confidence < self.manual_review_threshold or not names
        return SentencingResponse(
            predicted_accusations=predicted_accusations,
            sentence=sentence,
            key_sentencing_factors=extract_factors(fact),
            similar_cases=similar_cases,
            confidence=confidence,
            requires_manual_review=requires_review,
            warnings=warnings,
        )
