"""Pinecone vector store — the single place Pinecone's SDK is used.

This is a swap point: the rest of the system speaks only in ``chunk_id`` and
plain float vectors, never in Pinecone SDK objects. Vectors are stored keyed by
``chunk_id``; metadata is kept deliberately lean (``document_id``, ``page``,
``source_title``) because the authoritative chunk text always resolves from
SQLite — that way the two stores can never drift apart.

The index is created lazily as serverless on first use. Its dimension must match
the embedder's output dimension.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from pinecone import Pinecone, ServerlessSpec

from src.models.chunk import Chunk


class PineconeVectorStore:
    """Stores and queries dense vectors, keyed by ``chunk_id``.

    The ``client`` argument allows injecting a fake in tests; in normal use a
    real ``pinecone.Pinecone`` client is created from the API key.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        index_name: str = "hybrid-rag-docs",
        dimension: int = 1024,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        batch_size: int = 100,
        client: object | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._index_name = index_name
        self._dimension = dimension
        self._metric = metric
        self._cloud = cloud
        self._region = region
        self._batch_size = batch_size
        self._client = client or Pinecone(api_key=api_key)
        self._index: object | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def ensure_index(self) -> None:
        """Create the serverless index if it does not already exist.

        Idempotent: a no-op when the index is present. Creating an index is a
        real cloud operation, so this only runs against a live Pinecone client.
        """
        if not self._client.has_index(self._index_name):
            self._client.create_index(
                name=self._index_name,
                dimension=self._dimension,
                metric=self._metric,
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> int:
        """Upsert one vector per chunk, keyed by ``chunk_id``. Returns the count.

        ``chunks`` and ``vectors`` must align one-to-one; a mismatch is a
        programming error and raises rather than silently truncating.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must align"
            )
        if not chunks:
            return 0
        self.ensure_index()
        index = self._get_index()
        records = [
            {
                "id": chunk.chunk_id,
                "values": list(vector),
                "metadata": _lean_metadata(chunk),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for batch in _batched(records, self._batch_size):
            index.upsert(vectors=batch)
        return len(records)

    def query(self, vector: Sequence[float], top_n: int) -> list[tuple[str, float]]:
        """Return up to ``top_n`` ``(chunk_id, score)`` pairs, highest score first."""
        if top_n < 1:
            return []
        index = self._get_index()
        response = index.query(vector=list(vector), top_k=top_n, include_metadata=False)
        return [(match["id"], float(match["score"])) for match in _matches(response)]

    def _get_index(self) -> object:
        if self._index is None:
            self._index = self._client.Index(self._index_name)
        return self._index


def _lean_metadata(chunk: Chunk) -> dict:
    """Minimal metadata for Pinecone. Null values are omitted (Pinecone rejects
    them); the full record always lives in SQLite."""
    metadata = {"document_id": chunk.document_id, "source_title": chunk.source_title}
    if chunk.page is not None:
        metadata["page"] = chunk.page
    return metadata


def _matches(response: object) -> list:
    """Pinecone query responses expose matches by key and by attribute; support
    both so a plain fake and the real SDK object work the same way."""
    if isinstance(response, dict):
        return list(response.get("matches", []))
    return list(getattr(response, "matches", []))


def _batched(items: Sequence[dict], size: int) -> Iterator[Sequence[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
