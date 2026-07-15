"""Cohere reranking — the single place Cohere's rerank endpoint is used.

Reranking reorders a set of candidate chunks by true relevance to the query.
Two contracts matter here:

- **Chunk identity is preserved.** Documents are sent as plain text in candidate
  order; each result's ``index`` maps back to the source ``Chunk``, so the
  ``chunk_id`` spine survives reranking intact.
- **Failure is visible, and leaks nothing.** If the provider call fails, we raise
  a typed :class:`RerankError` with a generic, client-safe message. The original
  exception is chained (``from exc``) so it is available for server-side logs
  only — provider internals never reach the caller's surface. The system never
  pretends reranking succeeded; whether to degrade or abort is the caller's
  decision, made explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cohere

from src.models.chunk import Chunk


class RerankError(RuntimeError):
    """Raised when reranking fails. Carries a safe message; the underlying
    provider error is available via ``__cause__`` for server-side logging."""


@dataclass(frozen=True)
class RerankedChunk:
    """A chunk paired with its Cohere relevance score (normalized to [0, 1])."""

    chunk: Chunk
    relevance_score: float


class CohereReranker:
    """Reorders candidate chunks by relevance via Cohere Rerank.

    The ``client`` argument allows injecting a fake in tests; in normal use a
    real ``cohere.ClientV2`` is created from the API key.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "rerank-v4.0-pro",
        max_tokens_per_doc: int = 4096,
        timeout: float = 30.0,
        client: object | None = None,
    ) -> None:
        self._model = model
        self._max_tokens_per_doc = max_tokens_per_doc
        self._client = client or cohere.ClientV2(api_key=api_key, timeout=timeout)

    def rerank(
        self,
        query: str,
        candidates: Sequence[Chunk],
        top_n: int,
    ) -> list[RerankedChunk]:
        """Return up to ``top_n`` candidates reordered by relevance, best first.

        Raises:
            RerankError: the rerank call failed (message is client-safe).
        """
        if top_n < 1 or not candidates:
            return []

        documents = [chunk.text for chunk in candidates]
        try:
            response = self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(candidates)),
                max_tokens_per_doc=self._max_tokens_per_doc,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a safe, typed error
            raise RerankError("Reranking failed") from exc

        return [
            RerankedChunk(
                chunk=candidates[result.index],
                relevance_score=float(result.relevance_score),
            )
            for result in response.results
        ]
