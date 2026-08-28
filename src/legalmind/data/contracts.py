from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LegalLabels(BaseModel):
    accusations: list[str] = Field(default_factory=list)
    relevant_articles: list[int] = Field(default_factory=list)
    imprisonment_months: int | None = None
    life_imprisonment: bool = False
    death_penalty: bool = False


class CleaningMetadata(BaseModel):
    normalized: bool = True
    target_leakage_removed: bool = False
    leakage_types: list[str] = Field(default_factory=list)
    dedup_group_id: str
    fact_sha256: str


class LengthMetadata(BaseModel):
    characters: int
    tokens: int | None = None


class ProcessedCase(BaseModel):
    case_id: str
    source_dataset: str
    source_split: str
    fact: str
    labels: LegalLabels
    accusation_ids: list[int] = Field(default_factory=list)
    cleaning_metadata: CleaningMetadata
    length_metadata: LengthMetadata
    review_status: Literal["unreviewed", "manually_reviewed"] = "unreviewed"

    @field_validator("fact")
    @classmethod
    def fact_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("fact must not be empty")
        return value


class DatasetManifest(BaseModel):
    dataset_version: str
    source_dataset: str
    source_status: str
    source_sha256: str
    seed: int
    split_strategy: str
    sizes: dict[str, int]
    num_labels: int
    output_sha256: dict[str, str]
    quality: dict[str, int | float | str | bool | None]
