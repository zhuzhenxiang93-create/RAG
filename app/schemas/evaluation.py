"""Evaluation API schemas."""

from typing import Dict, Literal, Optional

from pydantic import BaseModel


class EvaluationRequest(BaseModel):
    benchmark: Literal["lite_v1"] = "lite_v1"


class EvaluationRunSummary(BaseModel):
    run_id: str
    status: Literal["completed"]
    benchmark: str
    question_count: int
    answerable_count: int
    retrieval_metrics: Dict
    qa_metrics: Dict
    artifact_path: str


class EvaluationArtifactResponse(BaseModel):
    run_id: str
    payload: Dict
