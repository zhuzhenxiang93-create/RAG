from __future__ import annotations

from legalmind.pipeline.contracts import (
    LegalCaseAnalysisRequest,
    LegalCaseAnalysisResponse,
)
from legalmind.pipeline.core import LegalMindPipeline


def create_app(pipeline: LegalMindPipeline):
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - optional production dependency
        raise RuntimeError("Install the 'api' optional dependency to run the HTTP API") from exc

    app = FastAPI(title="LegalMind Case Analysis Research API", version="0.3.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "pipeline": "legalmind-analyze"}

    @app.post("/v1/analyze", response_model=LegalCaseAnalysisResponse)
    def analyze_case(request: LegalCaseAnalysisRequest) -> LegalCaseAnalysisResponse:
        return pipeline.analyze(
            request.fact,
            accusations=request.accusations,
            top_k=request.top_k,
            as_of_date=request.as_of_date,
        )

    return app
