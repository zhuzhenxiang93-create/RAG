from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SentenceType(str, Enum):
    death = "death"
    life = "life"
    fixed_term = "fixed_term"
    detention = "detention"
    control = "control"
    exempt = "exempt"
    unknown = "unknown"


class FineStatus(str, Enum):
    missing = "missing"
    zero = "zero"
    positive = "positive"
    invalid = "invalid"
    not_applicable = "not_applicable"


class FineLabel(BaseModel):
    status: FineStatus
    amount: int | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> "FineLabel":
        if self.status is FineStatus.positive and (self.amount is None or self.amount <= 0):
            raise ValueError("positive fine requires amount > 0")
        if self.status is FineStatus.zero and self.amount != 0:
            raise ValueError("zero fine requires amount == 0")
        if self.status in {FineStatus.missing, FineStatus.not_applicable} and self.amount is not None:
            raise ValueError(f"{self.status.value} fine must not have an amount")
        if self.amount is not None and self.amount < 0:
            raise ValueError("negative fine is invalid")
        return self


class ProbationLabel(BaseModel):
    imposed: bool | None = None
    months: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_semantics(self) -> "ProbationLabel":
        if self.imposed is False and self.months is not None:
            raise ValueError("probation months require imposed=true")
        return self


class DataQuality(BaseModel):
    date_available: bool
    fine_available: bool
    sentence_type_reliable: bool
    target_leakage_detected: bool


class TaskMasks(BaseModel):
    sentencing_train: bool = False
    fine_binary: bool = False
    fine_amount: bool = False
    accusation_train: bool = False
    retrieval_corpus: bool = False
    retrieval_eval: bool = False
    robustness_eval: bool = False


class ProcessedCaseV3(BaseModel):
    case_id: str
    source: str
    source_case_id: str
    source_schema_version: str
    original_split: str | None = None
    judgment_date: date | None = None
    date_quality: Literal["verified", "parsed", "missing", "invalid"] = "missing"
    court: str | None = None
    fact: str = Field(min_length=1)
    accusations: list[str] = Field(default_factory=list)
    relevant_articles: list[str] = Field(default_factory=list)
    sentence_type: SentenceType
    imprisonment_months: int | None = Field(default=None, ge=0)
    probation: ProbationLabel
    fine: FineLabel
    death_penalty: bool = False
    life_imprisonment: bool = False
    defendants: list[str] = Field(default_factory=list)
    label_quality: str
    fine_quality: str
    sentence_type_quality: str
    privacy_status: Literal["redacted", "clear", "blocked"]
    leakage_status: Literal["clean", "cleaned", "blocked"]
    duplicate_group_id: str
    source_case_group_id: str
    task_masks: TaskMasks
    data_quality: DataQuality
    dataset_version: Literal["3.0.0"] = "3.0.0"

    @model_validator(mode="after")
    def validate_sentence(self) -> "ProcessedCaseV3":
        if self.sentence_type in {SentenceType.death, SentenceType.life}:
            if self.imprisonment_months is not None:
                raise ValueError("death/life must not be converted to months")
        if self.death_penalty != (self.sentence_type is SentenceType.death):
            raise ValueError("death_penalty conflicts with sentence_type")
        if self.life_imprisonment != (self.sentence_type is SentenceType.life):
            raise ValueError("life_imprisonment conflicts with sentence_type")
        return self


class StatuteRecord(BaseModel):
    document_id: str
    title: str
    article_number: str
    content: str
    promulgation_date: date | None
    effective_date: date
    expiry_date: date | None = None
    status: Literal["effective", "expired", "repealed", "future"]
    source_url: str
    retrieved_at: str
    checksum: str
