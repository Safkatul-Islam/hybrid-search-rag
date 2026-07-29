"""HTTP routes — thin. They validate, call the service, and map errors only.

No business logic lives here: the query pipeline is entirely inside
``QueryService``. Provider failures are translated to safe HTTP status codes;
no provider message, stack trace, or local path reaches the response body.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

from src.api.observability import logger
from src.api.rate_limit import enforce_rate_limit
from src.api.schemas import (
    CitationOut,
    HealthResponse,
    IngestJobAccepted,
    IngestJobResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
)
from src.api.security import require_api_key
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


@router.post(
    "/query",
    response_model=QueryResponse,
    # Rate limit first: an unauthenticated flood is rejected before any
    # credential comparison work is done.
    dependencies=[Depends(enforce_rate_limit), Depends(require_api_key)],
)
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


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
def ready(request: Request, response: Response) -> ReadinessResponse:
    """Report whether dependencies this process owns are usable.

    Deliberately makes **no provider calls**: they cost rate-limit budget, and a
    transient upstream blip should not take this instance out of rotation.
    """
    checks = {
        "chunk_store": _check_chunk_store(request),
        "bm25_index": getattr(request.app.state, "hybrid", None) is not None,
        "providers_configured": bool(request.app.state.providers_configured),
    }
    all_ready = all(checks.values())
    if not all_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(ready=all_ready, checks=checks)


def _check_chunk_store(request: Request) -> bool:
    try:
        request.app.state.query_service._store.all_chunks()  # noqa: SLF001
    except Exception:
        logger.exception("readiness: chunk store unreachable")
        return False
    return True


@router.post(
    "/ingest",
    response_model=IngestJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(enforce_rate_limit), Depends(require_api_key)],
)
def ingest(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
) -> IngestJobAccepted:
    """Accept a document and index it in the background.

    Validation that can be done cheaply — filename, type, size — happens here so
    an obviously bad upload fails fast with a real status code. Everything that
    touches providers is deferred; poll ``GET /ingest/{job_id}`` for the outcome.
    """
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
    if not content:
        # Cheap to detect and unambiguously wrong, so reject it now rather than
        # accepting a job that can only ever fail.
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")

    jobs = request.app.state.ingest_jobs
    job = jobs.create(source_title=name)
    logger.info("ingest accepted job_id=%s bytes=%d", job.job_id, len(content))

    background.add_task(
        _run_ingest,
        job_id=job.job_id,
        content=content,
        filename=name,
        jobs=jobs,
        indexing_service=request.app.state.indexing_service,
        hybrid=request.app.state.hybrid,
    )
    return IngestJobAccepted(
        job_id=job.job_id, status=job.status.value, source_title=job.source_title
    )


def _run_ingest(
    *, job_id: str, content: bytes, filename: str, jobs, indexing_service, hybrid
) -> None:
    """Index a document, recording the outcome on the job.

    Runs outside the request/response cycle, so nothing here may raise: an
    escaping exception would be lost and strand the job in ``running``.
    """
    jobs.mark_running(job_id)
    try:
        result = indexing_service.index_bytes(content, filename=filename)
    except ValueError:
        jobs.mark_failed(job_id, error="The file could not be read as a valid document.")
        logger.warning("ingest rejected job_id=%s reason=unreadable", job_id)
        return
    except Exception:
        # Provider or storage failure. The real cause goes to the log; the job
        # carries only a generic message, since clients can read it.
        logger.exception("ingest failed job_id=%s", job_id)
        jobs.mark_failed(job_id, error="Indexing failed. See server logs.")
        return

    try:
        # Make the new chunks searchable by keyword immediately.
        hybrid.update_bm25(indexing_service.rebuild_bm25())
    except Exception:
        logger.exception("bm25 refresh failed job_id=%s", job_id)
        jobs.mark_failed(job_id, error="Indexing failed. See server logs.")
        return

    jobs.mark_succeeded(
        job_id, document_id=result.document_id, chunk_count=result.chunk_count
    )
    logger.info(
        "ingest succeeded job_id=%s document_id=%s chunks=%d",
        job_id,
        result.document_id,
        result.chunk_count,
    )


@router.get(
    "/ingest/{job_id}",
    response_model=IngestJobResponse,
    dependencies=[Depends(enforce_rate_limit), Depends(require_api_key)],
)
def ingest_status(job_id: str, request: Request) -> IngestJobResponse:
    job = request.app.state.ingest_jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown job.")
    return IngestJobResponse(
        job_id=job.job_id,
        status=job.status.value,
        source_title=job.source_title,
        created_at=job.created_at,
        updated_at=job.updated_at,
        document_id=job.document_id,
        chunk_count=job.chunk_count,
        error=job.error,
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
