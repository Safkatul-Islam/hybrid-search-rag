"""HTTP routes — thin. They validate, call the service, and map errors only.

No business logic lives here: the query pipeline is entirely inside
``QueryService``. Provider failures are translated to safe HTTP status codes;
no provider message, stack trace, or local path reaches the response body.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from src.api.schemas import (
    CitationOut,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from src.ingestion.loader import SUPPORTED_SUFFIXES
from src.llm.client import LLMError
from src.services.query_service import QueryResult, QueryService

router = APIRouter()


def _query_service(request: Request) -> QueryService:
    return request.app.state.query_service


def _safe_basename(filename: str | None) -> str:
    """Reduce an untrusted upload filename to a bare basename.

    Both separators are normalized so a Windows-style path cannot slip a
    directory component through on a POSIX host. No filesystem path is ever
    built from this value — it is used only for format detection and display.
    """
    return PurePosixPath((filename or "").replace("\\", "/")).name.strip()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    service = _query_service(request)
    try:
        result = service.answer(body.question)
    except LLMError:
        # The underlying provider error is intentionally not surfaced.
        raise HTTPException(
            status_code=502, detail="Answer generation is temporarily unavailable."
        ) from None
    return _to_response(result)


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: Request, file: UploadFile = File(...)) -> IngestResponse:
    name = _safe_basename(file.filename)
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "A filename is required.")

    suffix = PurePosixPath(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type. Supported: {sorted(SUPPORTED_SUFFIXES)}",
        )

    max_bytes = request.app.state.max_upload_bytes
    content = file.file.read(max_bytes + 1)  # read one past the cap to detect overflow
    if len(content) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, "Uploaded file is too large."
        )

    indexing_service = request.app.state.indexing_service
    try:
        result = indexing_service.index_bytes(content, filename=name)
    except ValueError:
        # Loader rejected the content (empty / unreadable / no extractable text).
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "The file could not be read as a valid document.",
        ) from None

    # Make the new chunks searchable by keyword immediately.
    request.app.state.hybrid.update_bm25(indexing_service.rebuild_bm25())

    return IngestResponse(
        document_id=result.document_id,
        source_title=result.source_title,
        chunk_count=result.chunk_count,
    )


def _to_response(result: QueryResult) -> QueryResponse:
    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationOut(
                number=c.number,
                chunk_id=c.chunk_id,
                source_title=c.source_title,
                page=c.page,
            )
            for c in result.citations
        ],
        invalid_citation_numbers=result.invalid_citation_numbers,
        used_chunk_ids=result.used_chunk_ids,
        is_fallback=result.is_fallback,
        rerank_failed=result.rerank_failed,
    )
