"""Live Cohere embedding smoke test.

Skipped unless ``RUN_LIVE=1`` and a Cohere API key is configured. Run with:

    RUN_LIVE=1 pytest -m live
"""

from __future__ import annotations

import os

import pytest

from src.config import Settings

pytestmark = pytest.mark.live

_RUN_LIVE = os.getenv("RUN_LIVE") == "1"


@pytest.mark.skipif(not _RUN_LIVE, reason="RUN_LIVE != 1; live tests disabled")
def test_live_embed_returns_vectors_at_configured_dimension():
    settings = Settings()
    if not settings.cohere_api_key:
        pytest.skip("COHERE_API_KEY not set")

    from src.embeddings.embedder import CohereEmbedder

    embedder = CohereEmbedder(
        api_key=settings.cohere_api_key,
        model=settings.cohere_embed_model,
        dimension=settings.embed_dimension,
        timeout=settings.cohere_timeout,
    )

    doc_vectors = embedder.embed_documents(
        ["Hybrid search combines dense and keyword retrieval."]
    )
    query_vector = embedder.embed_query("What is hybrid search?")

    assert len(doc_vectors) == 1
    assert len(doc_vectors[0]) == settings.embed_dimension
    assert len(query_vector) == settings.embed_dimension
