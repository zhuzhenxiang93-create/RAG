"""Parser contracts and shared exceptions."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from app.schemas.documents import ParsedElement


class DocumentParseError(ValueError):
    """Raised when an accepted file cannot be parsed."""


class DocumentParser(ABC):
    """Format-specific parser interface."""

    @abstractmethod
    def parse(self, path: Path) -> List[ParsedElement]:
        """Extract location-aware elements from a document."""
