"""Dense retrieval — embed the query, then ask the vector store.

A thin coordinator over two isolated providers (the embedder and the vector
store). It holds no SDK code itself and returns plain ``(chunk_id, score)``
pairs, so the fusion layer never sees a provider object.
"""

from __future__ import annotations

from src.embeddings.embedder import CohereEmbedder
from src.storage.vector_store import PineconeVectorStore


class DenseRetriever:
    """Embeds a query and retrieves the nearest chunk vectors."""

    def __init__(
        self,
        *,
        embedder: CohereEmbedder,
        vector_store: PineconeVectorStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, query: str, top_n: int) -> list[tuple[str, float]]:
        """Return up to ``top_n`` ``(chunk_id, score)`` pairs, highest score first."""
        if top_n < 1:
            return []
        vector = self._embedder.embed_query(query)
        return self._vector_store.query(vector, top_n)
