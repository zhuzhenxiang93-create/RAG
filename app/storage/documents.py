"""SQLite metadata and JSON artifact storage."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.schemas.documents import DocumentChunk, DocumentDetail, DocumentRecord, ParsedElement


class DocumentRepository:
    """Persist document metadata while keeping parsed artifacts inspectable."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.db_path = data_dir / "docmind.db"
        self.artifact_dir = data_dir / "artifacts"

    def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(self.db_path))) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    element_count INTEGER NOT NULL,
                    parent_chunk_count INTEGER NOT NULL,
                    child_chunk_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    error TEXT,
                    stored_path TEXT NOT NULL
                )
                """
            )

    def save(
        self,
        record: DocumentRecord,
        stored_path: Path,
        elements: List[ParsedElement],
        chunks: List[DocumentChunk],
    ) -> None:
        self.initialize()
        artifact_path = self.artifact_dir / "{}.json".format(record.document_id)
        temporary = artifact_path.with_suffix(".json.part")
        temporary.write_text(
            json.dumps(
                {
                    "elements": [item.model_dump() for item in elements],
                    "chunks": [item.model_dump() for item in chunks],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(artifact_path)

        with closing(sqlite3.connect(str(self.db_path))) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.document_id,
                    record.filename,
                    record.file_type,
                    record.size_bytes,
                    record.sha256,
                    record.status,
                    record.element_count,
                    record.parent_chunk_count,
                    record.child_chunk_count,
                    record.created_at.isoformat(),
                    record.error,
                    str(stored_path),
                ),
            )

    def list(self) -> List[DocumentRecord]:
        self.initialize()
        with closing(sqlite3.connect(str(self.db_path))) as connection, connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [self._record(row) for row in rows]

    def get(self, document_id: str) -> Optional[DocumentDetail]:
        self.initialize()
        with closing(sqlite3.connect(str(self.db_path))) as connection, connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        if row is None:
            return None
        artifact = self.artifact_dir / "{}.json".format(document_id)
        payload = json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else {}
        record = self._record(row)
        return DocumentDetail(
            **record.model_dump(),
            elements=[ParsedElement(**item) for item in payload.get("elements", [])],
            chunks=[DocumentChunk(**item) for item in payload.get("chunks", [])],
        )

    def delete(self, document_id: str) -> bool:
        self.initialize()
        with closing(sqlite3.connect(str(self.db_path))) as connection, connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT stored_path FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            if row is None:
                return False
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        stored_path = Path(row["stored_path"])
        if stored_path.exists():
            stored_path.unlink()
        artifact = self.artifact_dir / "{}.json".format(document_id)
        if artifact.exists():
            artifact.unlink()
        return True

    @staticmethod
    def _record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            file_type=row["file_type"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            status=row["status"],
            element_count=row["element_count"],
            parent_chunk_count=row["parent_chunk_count"],
            child_chunk_count=row["child_chunk_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            error=row["error"],
        )
