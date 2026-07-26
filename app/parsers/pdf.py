"""Native-text PDF parser with page retention."""

from pathlib import Path
from typing import List
import warnings

from app.parsers.base import DocumentParseError, DocumentParser
from app.schemas.documents import ParsedElement


class PdfParser(DocumentParser):
    def parse(self, path: Path) -> List[ParsedElement]:
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="PyPDF2 is deprecated.*", category=DeprecationWarning
                    )
                    from PyPDF2 import PdfReader
            except ImportError as exc:
                raise DocumentParseError("PDF support requires pypdf") from exc

        try:
            reader = PdfReader(str(path))
            elements: List[ParsedElement] = []
            for page_number, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or "").strip()
                if text:
                    elements.append(
                        ParsedElement(
                            content=text,
                            page=page_number,
                            section="Page {}".format(page_number),
                            metadata={"source": "native_pdf"},
                        )
                    )
            if not elements:
                raise DocumentParseError(
                    "PDF contains no extractable text; enable OCR in Full mode"
                )
            return elements
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError("Failed to parse PDF: {}".format(exc)) from exc
