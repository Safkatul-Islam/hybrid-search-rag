"""PineconeVectorStore behavior, verified against a fake client (no network)."""

from __future__ import annotations

import pytest

from src.models.chunk import Chunk
from src.storage.vector_store import PineconeVectorStore
from tests.conftest import FakePineconeClient


def _chunk(chunk_id: str, position: int, *, page: int | None = None) -> Chunk:
    return Chunk(
        document_id="docabc",
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        source_title="src.pdf",
        position=position,
        page=page,
    )


def _store(client: FakePineconeClient, **kwargs) -> PineconeVectorStore:
    return PineconeVectorStore(client=client, dimension=3, **kwargs)


def test_upsert_keys_records_by_chunk_id_with_lean_metadata(fake_pinecone_client):
    store = _store(fake_pinecone_client)
    store.upsert([_chunk("docabc-0000", 0, page=5)], [[0.1, 0.2, 0.3]])

    record = fake_pinecone_client.index.vectors["docabc-0000"]
    assert record["values"] == [0.1, 0.2, 0.3]
    assert record["metadata"] == {
        "document_id": "docabc",
        "source_title": "src.pdf",
        "page": 5,
    }


def test_upsert_omits_null_page_from_metadata(fake_pinecone_client):
    store = _store(fake_pinecone_client)
    store.upsert([_chunk("docabc-0000", 0, page=None)], [[0.1, 0.2, 0.3]])

    assert "page" not in fake_pinecone_client.index.vectors["docabc-0000"]["metadata"]


def test_upsert_batches_over_the_configured_batch_size(fake_pinecone_client):
    store = _store(fake_pinecone_client, batch_size=100)
    chunks = [_chunk(f"docabc-{i:04d}", i) for i in range(250)]
    vectors = [[float(i), 0.0, 0.0] for i in range(250)]

    written = store.upsert(chunks, vectors)

    assert written == 250
    assert [len(batch) for batch in fake_pinecone_client.index.upsert_batches] == [
        100,
        100,
        50,
    ]


def test_upsert_rejects_misaligned_chunks_and_vectors(fake_pinecone_client):
    store = _store(fake_pinecone_client)
    with pytest.raises(ValueError):
        store.upsert([_chunk("docabc-0000", 0)], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


def test_query_returns_chunk_id_score_pairs_ranked(fake_pinecone_client):
    store = _store(fake_pinecone_client)
    store.upsert(
        [_chunk("docabc-0000", 0), _chunk("docabc-0001", 1)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    results = store.query([0.0, 1.0, 0.0], top_n=2)

    assert results[0][0] == "docabc-0001"
    assert [chunk_id for chunk_id, _ in results] == ["docabc-0001", "docabc-0000"]


def test_query_top_n_zero_returns_empty(fake_pinecone_client):
    store = _store(fake_pinecone_client)
    store.upsert([_chunk("docabc-0000", 0)], [[1.0, 0.0, 0.0]])
    assert store.query([1.0, 0.0, 0.0], top_n=0) == []


def test_ensure_index_creates_only_when_absent():
    client = FakePineconeClient(existing=False)
    store = PineconeVectorStore(client=client, dimension=3)
    store.ensure_index()
    store.ensure_index()  # idempotent
    assert len(client.created) == 1
    assert client.created[0]["dimension"] == 3
    assert client.created[0]["metric"] == "cosine"


def test_ensure_index_is_noop_when_index_exists():
    client = FakePineconeClient(existing=True)
    PineconeVectorStore(client=client, dimension=3).ensure_index()
    assert client.created == []
