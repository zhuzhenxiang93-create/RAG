"""Document lifecycle endpoints."""

from fastapi import APIRouter, File, Request, Response, UploadFile, status

from app.schemas.documents import DocumentDetail, DocumentListResponse, DocumentRecord
from app.services.documents import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


def _service(request: Request) -> DocumentService:
    settings = request.app.state.settings
    return DocumentService(settings.data_dir, settings.max_upload_mb)


@router.post("/upload", response_model=DocumentRecord, status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, file: UploadFile = File(...)) -> DocumentRecord:
    """Upload, parse and chunk one supported document."""
    try:
        return _service(request).ingest(file.filename or "", file.file)
    finally:
        await file.close()


@router.get("", response_model=DocumentListResponse)
async def list_documents(request: Request) -> DocumentListResponse:
    documents = _service(request).list_documents()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str, request: Request) -> DocumentDetail:
    return _service(request).get_document(document_id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, request: Request) -> Response:
    _service(request).delete_document(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
