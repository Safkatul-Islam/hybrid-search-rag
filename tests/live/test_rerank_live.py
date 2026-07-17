"""Live Cohere rerank smoke test.

Skipped unless ``RUN_LIVE=1`` and a Cohere API key is configured. Run with:

    RUN_LIVE=1 pytest -m live

Spends one real Cohere rerank call. Uses a query with one clearly-relevant
candidate so the expected top result is deterministic.
"""

from __future__ import annotations

import os

import pytest

from src.config import Settings
from src.models.chunk import Chunk

pytestmark = pytest.mark.live

_RUN_LIVE = os.getenv("RUN_LIVE") == "1"


def _chunk(chunk_id: str, text: str, position: int) -> Chunk:
    return Chunk(
        document_id="live",
        chunk_id=chunk_id,
        text=text,
        source_title="live.txt",
        position=position,
    )


@pytest.mark.skipif(not _RUN_LIVE, reason="RUN_LIVE != 1; live tests disabled")
def test_live_rerank_ranks_relevant_chunk_first():
    settings = Settings()
    if not settings.cohere_api_key:
        pytest.skip("COHERE_API_KEY not set")

    from src.reranking.reranker import CohereReranker

    reranker = CohereReranker(
        api_key=settings.cohere_api_key,
        model=settings.cohere_rerank_model,
        max_tokens_per_doc=settings.rerank_max_tokens_per_doc,
        timeout=settings.cohere_timeout,
    )

    candidates = [
        _chunk("c0", "The office cafeteria menu changes every Monday.", 0),
        _chunk("c1", "To reset your password, visit the account settings page.", 1),
        _chunk("c2", "Annual leave must be requested two weeks in advance.", 2),
    ]

    results = reranker.rerank("How do I reset my password?", candidates, top_n=3)

    assert results[0].chunk.chunk_id == "c1"
    assert all(0.0 <= r.relevance_score <= 1.0 for r in results)
