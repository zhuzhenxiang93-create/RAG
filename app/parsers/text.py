"""Plain-text and Markdown parsers."""

import re
from pathlib import Path
from typing import List, Optional

from app.parsers.base import DocumentParseError, DocumentParser
from app.schemas.documents import ParsedElement


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("Unable to decode text file as UTF-8 or GB18030")


class TextParser(DocumentParser):
    def parse(self, path: Path) -> List[ParsedElement]:
        content = _read_text(path).strip()
        return [ParsedElement(content=content)] if content else []


class MarkdownParser(DocumentParser):
    """Preserve Markdown heading hierarchy without rendering HTML."""

    _heading = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def parse(self, path: Path) -> List[ParsedElement]:
        lines = _read_text(path).splitlines()
        elements: List[ParsedElement] = []
        section: Optional[str] = None
        buffer: List[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            if text:
                elements.append(ParsedElement(content=text, section=section))
            buffer.clear()

        for line in lines:
            match = self._heading.match(line)
            if match:
                flush()
                section = match.group(2).strip()
                elements.append(
                    ParsedElement(
                        content=section,
                        content_type="heading",
                        section=section,
                        metadata={"heading_level": len(match.group(1))},
                    )
                )
            else:
                buffer.append(line)
        flush()
        return elements
