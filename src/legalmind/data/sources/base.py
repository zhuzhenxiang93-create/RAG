from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class SourceAdapter(ABC):
    name: str

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @abstractmethod
    def records(self) -> Iterable[dict]: ...
