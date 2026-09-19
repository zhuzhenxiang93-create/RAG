from __future__ import annotations

import hashlib
import re

from legalmind.schemas import CaseChunk, CaseRecord

_BOUNDARY_RE = re.compile(r"(?<=[。！？；\n])")


class ChineseCaseChunker:
    """Sentence-aware chunker with stable offsets and optional tokenizer budgeting."""

    def __init__(self, chunk_size: int = 448, overlap: int = 64, tokenizer=None):
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > overlap >= 0")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer

    def _length(self, text: str) -> int:
        if self.tokenizer is None:
            return len(text)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _largest_end(self, text: str, start: int) -> int:
        low, high = start + 1, len(text)
        best = low
        while low <= high:
            middle = (low + high) // 2
            if self._length(text[start:middle]) <= self.chunk_size:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    def _overlap_start(self, text: str, start: int, end: int) -> int:
        if self.overlap == 0:
            return end
        low, high = start + 1, end
        best = end
        while low <= high:
            middle = (low + high) // 2
            if self._length(text[middle:end]) <= self.overlap:
                best = middle
                high = middle - 1
            else:
                low = middle + 1
        return best

    def split(self, case: CaseRecord) -> list[CaseChunk]:
        text = case.fact
        if self._length(text) <= self.chunk_size:
            return [self._chunk(case, 0, 0, len(text))]
        boundaries = {match.end() for match in _BOUNDARY_RE.finditer(text)}
        chunks: list[CaseChunk] = []
        start = 0
        index = 0
        while start < len(text):
            limit = self._largest_end(text, start)
            candidates = [point for point in boundaries if start < point <= limit]
            end = max(candidates) if candidates else limit
            # Avoid producing a tiny chunk when the nearest sentence boundary is far behind.
            if candidates and self._length(text[start:end]) < self.chunk_size // 2:
                end = limit
            if end <= start:
                end = limit
            chunks.append(self._chunk(case, index, start, end))
            if end == len(text):
                break
            next_start = self._overlap_start(text, start, end)
            start = next_start if next_start > start else end
            index += 1
        return chunks

    @staticmethod
    def _chunk(case: CaseRecord, index: int, start: int, end: int) -> CaseChunk:
        return CaseChunk(
            chunk_id=f"{case.case_id}:c{index}",
            case_id=case.case_id,
            text=case.fact[start:end],
            chunk_index=index,
            start_char=start,
            end_char=end,
            accusations=case.accusations,
            relevant_articles=case.relevant_articles,
            penalty=case.penalty,
            source_split=case.source_split,
            parent_title="案件事实",
            text_sha256=hashlib.sha256(case.fact[start:end].encode("utf-8")).hexdigest(),
        )
