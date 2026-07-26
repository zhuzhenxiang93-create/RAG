"""Secure document ingestion orchestration."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, List
from uuid import uuid4

from app.chunking.parent_child import ParentChildChunker
from app.core.exceptions import AppError
from app.parsers.base import DocumentParseError
from app.parsers.registry import ParserRegistry
from app.schemas.documents import DocumentDetail, DocumentRecord
from app.storage.documents import DocumentRepository


class DocumentService:
    """Validate, store, parse, chunk and persist uploaded documents."""

    def __init__(self, data_dir: Path, max_upload_mb: int) -> None:
        self.data_dir = data_dir
        self.max_bytes = max_upload_mb * 1024 * 1024
        self.registry = ParserRegistry()
        self.chunker = ParentChildChunker()
        self.repository = DocumentRepository(data_dir)

    def ingest(self, original_filename: str, stream: BinaryIO) -> DocumentRecord:
        filename = self._safe_filename(original_filename)
        extension = Path(filename).suffix.lower()
        if extension not in self.registry.allowed_extensions:
            raise AppError(
                "Unsupported file type '{}'; allowed: {}".format(
                    extension or "(none)", ", ".join(self.registry.allowed_extensions)
                ),
                code="unsupported_file_type",
                status_code=415,
            )

        document_id = uuid4().hex
        upload_dir = self.data_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        final_path = upload_dir / "{}{}".format(document_id, extension)
        temporary = final_path.with_suffix(extension + ".part")
        digest = hashlib.sha256()
        size = 0

        try:
            with temporary.open("wb") as destination:
                while True:
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > self.max_bytes:
                        raise AppError(
                            "Upload exceeds {} MB limit".format(
                                self.max_bytes // (1024 * 1024)
                            ),
                            code="file_too_large",
                            status_code=413,
                        )
                    digest.update(block)
                    destination.write(block)
            temporary.replace(final_path)
            elements = self.registry.parse(final_path)
            if not elements:
                raise DocumentParseError("Document contains no usable content")
            chunks = self.chunker.chunk(document_id, elements)
            record = DocumentRecord(
                document_id=document_id,
                filename=filename,
                file_type=extension.lstrip("."),
                size_bytes=size,
                sha256=digest.hexdigest(),
                status="ready",
                element_count=len(elements),
                parent_chunk_count=sum(item.level == "parent" for item in chunks),
                child_chunk_count=sum(item.level == "child" for item in chunks),
                created_at=datetime.now(timezone.utc),
            )
            self.repository.save(record, final_path, elements, chunks)
            return record
        except AppError:
            self._cleanup(temporary, final_path)
            raise
        except DocumentParseError as exc:
            self._cleanup(temporary, final_path)
            raise AppError(str(exc), code="document_parse_failed", status_code=422) from exc
        except Exception as exc:
            self._cleanup(temporary, final_path)
            raise AppError(
                "Document ingestion failed",
                code="document_ingestion_failed",
                status_code=500,
            ) from exc

    def list_documents(self) -> List[DocumentRecord]:
        return self.repository.list()

    def get_document(self, document_id: str) -> DocumentDetail:
        detail = self.repository.get(document_id)
        if detail is None:
            raise AppError("Document not found", code="document_not_found", status_code=404)
        return detail

    def delete_document(self, document_id: str) -> None:
        if not self.repository.delete(document_id):
            raise AppError("Document not found", code="document_not_found", status_code=404)

    @staticmethod
    def _safe_filename(value: str) -> str:
        filename = Path(value or "").name.strip()
        if not filename or filename in {".", ".."}:
            raise AppError("A valid filename is required", code="invalid_filename")
        if any(character in filename for character in ("\x00", "\r", "\n")):
            raise AppError("Filename contains invalid characters", code="invalid_filename")
        return filename

    @staticmethod
    def _cleanup(*paths: Path) -> None:
        for path in paths:
            if path.exists():
                path.unlink()
