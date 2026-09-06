"""Structured, evidence-aware sentencing research baseline."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legalmind.sentencing.model import SentencingBaseline
    from legalmind.sentencing.service import SentencingService

__all__ = ["SentencingBaseline", "SentencingService"]


def __getattr__(name: str):
    if name == "SentencingBaseline":
        from legalmind.sentencing.model import SentencingBaseline

        return SentencingBaseline
    if name == "SentencingService":
        from legalmind.sentencing.service import SentencingService

        return SentencingService
    raise AttributeError(name)
