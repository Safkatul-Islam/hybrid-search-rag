"""Request/response models for the HTTP boundary.

Validation lives here so malformed input is rejected before any provider is
touched. Response models are explicit (not raw dicts) so the shape returned to
clients is stable and never leaks internal objects.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.config import get_settings

# Resolved once at import; the boundary length cap comes from central config.
_MAX_QUESTION_LENGTH = get_settings().max_query_length


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=_MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class CitationOut(BaseModel):
    number: int
    chunk_id: str
    source_title: str
    page: int | None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    invalid_citation_numbers: list[int]
    used_chunk_ids: list[str]
    is_fallback: bool
    rerank_failed: bool


class HealthResponse(BaseModel):
    status: str


class IngestResponse(BaseModel):
    document_id: str
    source_title: str
    chunk_count: int
