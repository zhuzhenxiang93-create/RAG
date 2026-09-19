from typing import TYPE_CHECKING, Any

from legalmind.pipeline.contracts import LegalCaseAnalysisResponse

if TYPE_CHECKING:
    from legalmind.pipeline.core import LegalMindPipeline

__all__ = ["LegalCaseAnalysisResponse", "LegalMindPipeline"]


def __getattr__(name: str) -> Any:
    """Load the pipeline implementation lazily to avoid package import cycles."""
    if name == "LegalMindPipeline":
        from legalmind.pipeline.core import LegalMindPipeline

        return LegalMindPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
