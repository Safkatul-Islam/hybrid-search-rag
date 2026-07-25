"""Query service — the whole pipeline as a single call.

Flow: hybrid retrieve -> resolve fused chunk_ids to Chunks (SQLite is the source
of truth) -> rerank -> generate a grounded, cited answer.

Two policies decided here (see DECISIONS #18):

- **Rerank failure degrades, visibly.** If reranking raises, fall back to the
  fusion-ranked top-N and set ``rerank_failed=True`` — a useful answer is still
  returned, but the failure is surfaced, never pretended away.
- **Hallucinated citations are flagged, not rejected.** Valid citations resolve;
  out-of-range numbers surface in ``invalid_citation_numbers``. The answer is
  returned as-is for the caller to act on.

An empty query short-circuits to the safe fallback without touching any provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.generation.citations import Citation
from src.generation.generator import RagGenerator
from src.models.chunk import Chunk
from src.prompts.templates import FALLBACK_ANSWER
from src.reranking.reranker import CohereReranker, RerankError
from src.retrieval.hybrid import HybridRetriever
from src.storage.chunk_store import ChunkStore


@dataclass(frozen=True)
class QueryResult:
    """The answer to a query plus provenance and observability flags."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    invalid_citation_numbers: list[int] = field(default_factory=list)
    used_chunk_ids: list[str] = field(default_factory=list)
    is_fallback: bool = False
    rerank_failed: bool = False


class QueryService:
    """Answers a question over the indexed corpus, end to end."""

    def __init__(
        self,
        *,
        hybrid: HybridRetriever,
        store: ChunkStore,
        reranker: CohereReranker,
        generator: RagGenerator,
        rerank_top_n: int,
    ) -> None:
        self._hybrid = hybrid
        self._store = store
        self._reranker = reranker
        self._generator = generator
        self._rerank_top_n = rerank_top_n

    def answer(self, query: str) -> QueryResult:
        """Return a grounded, cited answer for ``query``."""
        if not query.strip():
            return QueryResult(answer=FALLBACK_ANSWER, is_fallback=True)

        fused = self._hybrid.retrieve(query)
        candidates = self._store.get_chunks([chunk_id for chunk_id, _ in fused])
        if not candidates:
            return QueryResult(answer=FALLBACK_ANSWER, is_fallback=True)

        ranked, rerank_failed = self._rerank(query, candidates)
        generated = self._generator.generate(query, ranked)

        return QueryResult(
            answer=generated.text,
            citations=generated.citations,
            invalid_citation_numbers=generated.invalid_citation_numbers,
            used_chunk_ids=[chunk.chunk_id for chunk in ranked],
            is_fallback=generated.is_fallback,
            rerank_failed=rerank_failed,
        )

    def _rerank(
        self, query: str, candidates: list[Chunk]
    ) -> tuple[list[Chunk], bool]:
        """Rerank candidates; on failure, degrade to fusion order (flagged)."""
        try:
            reranked = self._reranker.rerank(query, candidates, self._rerank_top_n)
        except RerankError:
            return candidates[: self._rerank_top_n], True
        return [item.chunk for item in reranked], False
