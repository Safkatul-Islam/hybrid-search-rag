"""Live Pinecone vector-store smoke test.

Skipped unless ``RUN_LIVE=1`` and a Pinecone API key is configured. Run with:

    RUN_LIVE=1 pytest -m live

This test creates the serverless index (if absent) in the configured Pinecone
account and leaves two synthetic vectors behind — a real cloud side effect, which
is why it is gated. It uses synthetic vectors (not Cohere) so it exercises only
the Pinecone wiring. Serverless upserts are eventually consistent, so the query
is retried briefly.
"""

from __future__ import annotations

import os
import time

import pytest

from src.config import Settings
from src.models.chunk import Chunk

pytestmark = pytest.mark.live

_RUN_LIVE = os.getenv("RUN_LIVE") == "1"


def _synthetic_chunk(chunk_id: str, position: int) -> Chunk:
    return Chunk(
        document_id="live-smoke",
        chunk_id=chunk_id,
        text=f"live smoke chunk {position}",
        source_title="live-smoke.txt",
        position=position,
        page=None,
    )


@pytest.mark.skipif(not _RUN_LIVE, reason="RUN_LIVE != 1; live tests disabled")
def test_live_upsert_then_query_returns_seeded_chunk():
    settings = Settings()
    if not settings.pinecone_api_key:
        pytest.skip("PINECONE_API_KEY not set")

    from src.storage.vector_store import PineconeVectorStore

    dim = settings.embed_dimension
    store = PineconeVectorStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        dimension=dim,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )

    # Two distinct unit vectors along different axes.
    vec_a = [1.0] + [0.0] * (dim - 1)
    vec_b = [0.0, 1.0] + [0.0] * (dim - 2)
    chunks = [_synthetic_chunk("live-smoke-0000", 0), _synthetic_chunk("live-smoke-0001", 1)]
    store.upsert(chunks, [vec_a, vec_b])

    # Serverless upserts are eventually consistent: retry the query briefly.
    top_id = None
    for _ in range(15):
        results = store.query(vec_a, top_n=2)
        if results:
            top_id = results[0][0]
            break
        time.sleep(2)

    assert top_id == "live-smoke-0000"
