"""Heading-aware Parent-Child chunking."""

import hashlib
from typing import Iterable, List, Optional

from app.schemas.documents import DocumentChunk, ParsedElement


class ParentChildChunker:
    """Create precise child chunks backed by larger context chunks."""

    def __init__(
        self,
        parent_size: int = 1800,
        child_size: int = 520,
        child_overlap: int = 80,
    ) -> None:
        if child_overlap >= child_size:
            raise ValueError("child_overlap must be smaller than child_size")
        if parent_size < child_size:
            raise ValueError("parent_size must be at least child_size")
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk(
        self, document_id: str, elements: Iterable[ParsedElement]
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        parent_index = 0

        for group in self._groups(elements):
            text = "\n\n".join(item.content.strip() for item in group if item.content.strip())
            if not text:
                continue
            first = group[0]
            for parent_text in self._windows(text, self.parent_size, overlap=0):
                parent_index += 1
                parent_id = self._id(document_id, "parent", parent_index, parent_text)
                parent = DocumentChunk(
                    chunk_id=parent_id,
                    document_id=document_id,
                    level="parent",
                    content=parent_text,
                    page=first.page,
                    section=first.section,
                    content_type=first.content_type,
                    metadata={"element_count": len(group)},
                )
                chunks.append(parent)

                child_index = 0
                for child_text in self._windows(
                    parent_text, self.child_size, overlap=self.child_overlap
                ):
                    child_index += 1
                    searchable = self._with_section(first.section, child_text)
                    child_id = self._id(
                        document_id,
                        "child",
                        parent_index * 100000 + child_index,
                        searchable,
                    )
                    chunks.append(
                        DocumentChunk(
                            chunk_id=child_id,
                            document_id=document_id,
                            level="child",
                            parent_id=parent_id,
                            content=searchable,
                            page=first.page,
                            section=first.section,
                            content_type=first.content_type,
                            metadata={"raw_content": child_text},
                        )
                    )
        return chunks

    @staticmethod
    def _groups(elements: Iterable[ParsedElement]) -> List[List[ParsedElement]]:
        groups: List[List[ParsedElement]] = []
        current: List[ParsedElement] = []
        key: Optional[tuple] = None
        for element in elements:
            element_key = (element.section, element.page, element.content_type == "table")
            if current and element_key != key:
                groups.append(current)
                current = []
            current.append(element)
            key = element_key
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _windows(text: str, size: int, overlap: int) -> List[str]:
        if len(text) <= size:
            return [text]
        result: List[str] = []
        start = 0
        step = size - overlap
        while start < len(text):
            end = min(start + size, len(text))
            candidate = text[start:end].strip()
            if candidate:
                result.append(candidate)
            if end == len(text):
                break
            start += step
        return result

    @staticmethod
    def _with_section(section: Optional[str], content: str) -> str:
        if not section or content.startswith(section):
            return content
        return "[Section: {}]\n{}".format(section, content)

    @staticmethod
    def _id(document_id: str, level: str, index: int, content: str) -> str:
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]
        return "{}-{}-{}-{}".format(document_id, level, index, digest)
