"""Hybrid retrieval — dense + BM25, combined with reciprocal rank fusion.

Composes the two retrievers over the shared ``chunk_id`` space and fuses their
rankings. Only rank order crosses into fusion (see ``fusion.py``); the raw dense
and BM25 scores are never mixed. Returns ranked ``chunk_id``s; resolving them to
full ``Chunk`` text is the caller's job (done from SQLite downstream).
"""

from __future__ import annotations

from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever:
    """Runs dense and BM25 retrieval and fuses the two rankings."""

    def __init__(
        self,
        *,
        dense: DenseRetriever,
        bm25: BM25Index,
        dense_top_n: int,
        bm25_top_n: int,
        rrf_k: int,
        fusion_top_n: int,
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._dense_top_n = dense_top_n
        self._bm25_top_n = bm25_top_n
        self._rrf_k = rrf_k
        self._fusion_top_n = fusion_top_n

    def retrieve(self, query: str) -> list[tuple[str, float]]:
        """Return the fused ``(chunk_id, fused_score)`` ranking for a query."""
        dense_hits = self._dense.retrieve(query, self._dense_top_n)
        bm25_hits = self._bm25.query(query, self._bm25_top_n)
        dense_ids = [chunk_id for chunk_id, _ in dense_hits]
        bm25_ids = [chunk_id for chunk_id, _ in bm25_hits]
        return reciprocal_rank_fusion(
            [dense_ids, bm25_ids],
            k=self._rrf_k,
            top_n=self._fusion_top_n,
        )
