"""Optional legal classification plugin schemas."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LegalPluginStatus(BaseModel):
    enabled: bool
    state: Literal["disabled", "not_configured", "dependencies_missing", "ready", "loaded"]
    dependencies: dict
    configured_assets: dict
    label_count: Optional[int] = None
    device: Optional[str] = None
    message: str


class LegalClassificationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    top_k: int = Field(default=3, ge=1, le=20)


class LegalPrediction(BaseModel):
    label: str
    label_id: int
    probability: float = Field(ge=0.0, le=1.0)


class LegalClassificationResponse(BaseModel):
    predictions: List[LegalPrediction]
    model: str
    adapter: str
    device: str
