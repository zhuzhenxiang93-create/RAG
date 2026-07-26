"""Safe parser dispatch."""

from pathlib import Path
from typing import Dict, List

from app.parsers.base import DocumentParseError, DocumentParser
from app.parsers.office import DocxParser, XlsxParser
from app.parsers.pdf import PdfParser
from app.parsers.text import MarkdownParser, TextParser
from app.schemas.documents import ParsedElement


class ParserRegistry:
    """Resolve a parser by normalized extension."""

    def __init__(self) -> None:
        self._parsers: Dict[str, DocumentParser] = {
            ".pdf": PdfParser(),
            ".docx": DocxParser(),
            ".xlsx": XlsxParser(),
            ".txt": TextParser(),
            ".md": MarkdownParser(),
        }

    @property
    def allowed_extensions(self) -> List[str]:
        return sorted(self._parsers)

    def parse(self, path: Path) -> List[ParsedElement]:
        extension = path.suffix.lower()
        parser = self._parsers.get(extension)
        if parser is None:
            raise DocumentParseError(
                "Unsupported file type '{}'; allowed: {}".format(
                    extension or "(none)", ", ".join(self.allowed_extensions)
                )
            )
        return parser.parse(path)
