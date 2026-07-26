"""Built-in reproducible evaluation endpoints."""

from fastapi import APIRouter, Request

from app.core.exceptions import AppError
from app.evaluation.runner import EvaluationRunner
from app.schemas.evaluation import (
    EvaluationArtifactResponse,
    EvaluationRequest,
    EvaluationRunSummary,
)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _runner(request: Request) -> EvaluationRunner:
    settings = request.app.state.settings
    return EvaluationRunner(
        settings.project_root,
        settings.project_root / "artifacts" / "evaluation",
    )


@router.post("/run", response_model=EvaluationRunSummary)
async def run_evaluation(
    payload: EvaluationRequest, request: Request
) -> EvaluationRunSummary:
    return _runner(request).run(payload.benchmark)


@router.get("/{run_id}", response_model=EvaluationArtifactResponse)
async def get_evaluation(run_id: str, request: Request) -> EvaluationArtifactResponse:
    try:
        payload = _runner(request).load(run_id)
    except (ValueError, FileNotFoundError) as exc:
        raise AppError(
            "Evaluation run not found",
            code="evaluation_not_found",
            status_code=404,
        ) from exc
    return EvaluationArtifactResponse(run_id=run_id, payload=payload)
