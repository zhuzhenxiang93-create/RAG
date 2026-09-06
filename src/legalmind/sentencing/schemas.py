from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictSentencingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SentenceType(str, Enum):
    death = "death"
    life = "life"
    fixed_term = "fixed_term"
    detention = "detention"
    control = "control"
    exempt = "exempt"
    unknown = "unknown"


class PredictedAccusation(StrictSentencingModel):
    name: str
    probability: float = Field(ge=0.0, le=1.0)


class SentencingRequest(StrictSentencingModel):
    fact: str = ""
    accusation: list[str] = Field(default_factory=list)
    predicted_accusations: list[PredictedAccusation] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=0, le=3)


class NumericRange(StrictSentencingModel):
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> NumericRange:
        if self.min > self.max:
            raise ValueError("range min must not exceed max")
        return self


class MonthsRange(StrictSentencingModel):
    min_months: int = Field(ge=0)
    max_months: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> MonthsRange:
        if self.min_months > self.max_months:
            raise ValueError("min_months must not exceed max_months")
        return self


class BinaryPrediction(StrictSentencingModel):
    predicted: bool | None = None
    probability: float | None = Field(default=None, ge=0.0, le=1.0)


class FinePrediction(StrictSentencingModel):
    imposed: bool | None = None
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    amount: int | None = Field(default=None, ge=0)
    amount_range: NumericRange | None = None

    @model_validator(mode="after")
    def amount_requires_positive_status(self) -> FinePrediction:
        if (self.amount is not None or self.amount_range is not None) and self.imposed is not True:
            raise ValueError("a fine amount requires imposed=true")
        return self


class SentencePrediction(StrictSentencingModel):
    sentence_type: SentenceType = SentenceType.unknown
    sentence_type_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    imprisonment_months: int | None = Field(default=None, ge=0)
    imprisonment_range: MonthsRange | None = None
    probation: BinaryPrediction = Field(default_factory=BinaryPrediction)
    fine: FinePrediction = Field(default_factory=FinePrediction)
    death_penalty: BinaryPrediction = Field(default_factory=BinaryPrediction)
    life_imprisonment: BinaryPrediction = Field(default_factory=BinaryPrediction)

    @model_validator(mode="after")
    def preserve_sentence_semantics(self) -> SentencePrediction:
        non_month_sentence = self.sentence_type in {
            SentenceType.death,
            SentenceType.life,
            SentenceType.exempt,
        }
        if non_month_sentence and (
            self.imprisonment_months is not None or self.imprisonment_range is not None
        ):
            raise ValueError("death, life, and exempt sentences must not use imprisonment months")
        if self.fine.imposed is False and (
            self.fine.amount is not None or self.fine.amount_range is not None
        ):
            raise ValueError("a negative fine prediction must not include an amount")
        if self.death_penalty.predicted is True and self.sentence_type is not SentenceType.death:
            raise ValueError("death_penalty=true requires sentence_type=death")
        if self.life_imprisonment.predicted is True and self.sentence_type is not SentenceType.life:
            raise ValueError("life_imprisonment=true requires sentence_type=life")
        return self


class SimilarCase(StrictSentencingModel):
    case_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    accusations: list[str] = Field(default_factory=list)
    sentence_type: SentenceType
    imprisonment_months: int | None = Field(default=None, ge=0)
    fine: int | None = Field(default=None, ge=0)
    summary: str


class SentencingResponse(StrictSentencingModel):
    status: str = "ok"
    predicted_accusations: list[PredictedAccusation] = Field(default_factory=list)
    sentence: SentencePrediction = Field(default_factory=SentencePrediction)
    key_sentencing_factors: list[str] = Field(default_factory=list)
    similar_cases: list[SimilarCase] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_manual_review: bool = True
    missing_information: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = "该结果来自历史案件数据模型，仅用于研究和辅助分析，不构成法律意见。"
