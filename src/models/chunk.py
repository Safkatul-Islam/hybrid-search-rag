"""The Chunk model and deterministic identity helpers.

Chunk identity is the spine of the whole system: the same ``chunk_id`` must be
usable across SQLite, the vector store, BM25, reranking, and citations. IDs are
derived deterministically so that re-ingesting the same document yields the same
identities (an upsert), rather than duplicates.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

# Length of the hex prefix used for document ids. 16 hex chars = 64 bits, which
# is collision-safe at the document scale this MVP targets while staying short
# enough to be readable when inspecting the store.
DOCUMENT_ID_LENGTH = 16


def compute_document_id(content: bytes) -> str:
    """Deterministic id derived from raw file bytes.

    Identical content always produces the same id; changed content produces a
    different one (so an edited document is treated as a new document).
    """
    return hashlib.sha256(content).hexdigest()[:DOCUMENT_ID_LENGTH]


def make_chunk_id(document_id: str, position: int) -> str:
    """Deterministic, position-stable chunk id for a document."""
    return f"{document_id}-{position:04d}"


class Chunk(BaseModel):
    """An immutable unit of retrievable text with stable identity and provenance."""

    model_config = {"frozen": True}

    document_id: str
    chunk_id: str
    text: str
    source_title: str
    position: int
    page: int | None = None
    section: str | None = None
    # Open bag for future needs (e.g. access control). Not interpreted yet.
    metadata: dict = Field(default_factory=dict)
