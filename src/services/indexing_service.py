"""Indexing service — orchestrates the ingestion pipeline end to end.

One document flows: load -> chunk -> SQLite (source of truth) -> embed ->
Pinecone. SQLite is written before the vector store so the canonical record
exists even if embedding or upsert fails. BM25 is rebuilt from SQLite on demand,
never from the in-flight chunks, so the keyword index can never drift from the
canonical store.

Collaborators are injected, so tests can supply fakes and providers can be
swapped without touching this logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.embeddings.embedder import CohereEmbedder
from src.ingestion.chunker import chunk_document
from src.ingestion.loader import LoadedDocument, load, load_bytes
from src.retrieval.bm25 import BM25Index
from src.storage.chunk_store import ChunkStore
from src.storage.vector_store import PineconeVectorStore


@dataclass(frozen=True)
class IndexingResult:
    """Outcome of indexing a single document."""

    document_id: str
    source_title: str
    chunk_count: int


class IndexingService:
    """Ties ingestion, storage, embedding, and the vector store together."""

    def __init__(
        self,
        *,
        store: ChunkStore,
        embedder: CohereEmbedder,
        vector_store: PineconeVectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def index_document(self, path: str | Path) -> IndexingResult:
        """Load, chunk, persist, embed, and upsert one document from disk."""
        return self._index(load(path))

    def index_bytes(self, content: bytes, *, filename: str) -> IndexingResult:
        """Same pipeline as ``index_document`` but from in-memory upload bytes.

        ``filename`` is used only for format detection and as the display title;
        callers must pass a sanitized basename (never a path).
        """
        return self._index(load_bytes(content, source_title=filename))

    def _index(self, document: LoadedDocument) -> IndexingResult:
        chunks = chunk_document(
            document,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        if not chunks:
            return IndexingResult(
                document_id=document.document_id,
                source_title=document.source_title,
                chunk_count=0,
            )

        # SQLite first: the canonical record must exist before vectors do.
        self._store.upsert_chunks(chunks)
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        self._vector_store.upsert(chunks, vectors)

        return IndexingResult(
            document_id=document.document_id,
            source_title=document.source_title,
            chunk_count=len(chunks),
        )

    def rebuild_bm25(self) -> BM25Index:
        """Build a fresh BM25 index from every chunk in the canonical store."""
        return BM25Index(self._store.all_chunks())
