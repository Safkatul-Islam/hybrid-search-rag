"""Request-scoped logging.

Every request gets an id, echoed back as ``X-Request-ID`` and attached to each
log line emitted while handling it, so a report of "my query failed" can be
traced through the pipeline without correlating timestamps.

**What is never logged:** document text, chunk text, answers, questions, the API
key, and provider keys. Log lines carry identifiers, counts, and outcomes only.
This is enforced by test, not just convention — see ``tests/test_observability``.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

logger = logging.getLogger("hybrid_rag")

#: Marks handlers installed by ``configure_logging`` so repeat calls replace
#: only those, leaving anything else attached to the logger alone.
_OWNED_MARKER = "_hybrid_rag_owned"


def current_request_id() -> str:
    """The id of the request being handled, or ``"-"`` outside a request."""
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Injects ``request_id`` into every record so the format string can use it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Install a single stream handler carrying the request id.

    Idempotent: repeated calls (one per app construction, and tests build many)
    must not stack duplicate handlers.
    """
    root = logging.getLogger("hybrid_rag")
    root.setLevel(level.upper())
    # Own handler, no propagation: uvicorn configures the root logger, and
    # propagating would print every line twice in a different format.
    root.propagate = False
    # Remove only handlers this function installed. Tearing off every handler
    # would silently detach anything else attached to this logger — a log
    # aggregator, or a test collecting records.
    for handler in list(root.handlers):
        if getattr(handler, _OWNED_MARKER, False):
            root.removeHandler(handler)

    # The filter belongs on the logger, not the handler, so ``request_id`` is
    # present on every record this logger emits — including for any handler a
    # caller or test attaches later.
    if not any(isinstance(f, RequestIdFilter) for f in root.filters):
        root.addFilter(RequestIdFilter())

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(message)s")
    )
    setattr(handler, _OWNED_MARKER, True)
    root.addHandler(handler)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, log the outcome, and echo the id back."""

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        # Accept a caller-supplied id for tracing across hops, but bound its
        # length and character set: it is untrusted input that lands in logs.
        request_id = (
            incoming[:64]
            if incoming and incoming.replace("-", "").replace("_", "").isalnum()
            else uuid.uuid4().hex[:16]
        )
        token = _request_id.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            # Path and method only -- never the body.
            logger.exception(
                "unhandled error method=%s path=%s elapsed_ms=%.1f",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            _request_id.reset(token)
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "method=%s path=%s status=%d elapsed_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers[REQUEST_ID_HEADER] = request_id
        _request_id.reset(token)
        return response
