"""General intent classification and knowledge-route schemas."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class IntentPluginStatus(BaseModel):
    enabled: bool
    backend: Literal["lite", "lora"]
    state: Literal[
        "disabled",
        "lite_ready",
        "not_configured",
        "dependencies_missing",
        "ready",
        "loaded",
    ]
    dependencies: Dict[str, bool]
    configured_assets: Dict[str, bool]
    label_count: Optional[int] = None
    device: Optional[str] = None
    message: str


class IntentClassificationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    top_k: int = Field(default=3, ge=1, le=20)


class IntentPrediction(BaseModel):
    intent: str
    domain: str
    label_id: int
    probability: float = Field(ge=0.0, le=1.0)


class KnowledgeRoute(BaseModel):
    knowledge_base: str
    retrieval_profile: Literal["exact", "balanced", "semantic", "broad"]
    allow_workflow: bool = False
    abstain: bool = False
    reason: str


class IntentClassificationResponse(BaseModel):
    predictions: List[IntentPrediction]
    route: KnowledgeRoute
    backend: Literal["lite", "lora"]
    model: str
    adapter: Optional[str] = None
    device: str
    calibrated: bool = False

