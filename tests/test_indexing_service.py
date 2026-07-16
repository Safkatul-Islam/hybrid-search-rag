"""IndexingService orchestration, verified end to end against fakes (no network)."""

from __future__ import annotations

from pathlib import Path

from src.embeddings.embedder import CohereEmbedder
from src.services.indexing_service import IndexingService
from src.storage.chunk_store import ChunkStore
from src.storage.vector_store import PineconeVectorStore
from tests.conftest import FakeCohereClient, FakePineconeClient


def _service(
    tmp_path: Path,
    cohere: FakeCohereClient,
    pinecone: FakePineconeClient,
) -> tuple[IndexingService, ChunkStore]:
    store = ChunkStore(tmp_path / "chunks.sqlite")
    embedder = CohereEmbedder(client=cohere, dimension=8)
    vector_store = PineconeVectorStore(client=pinecone, dimension=8)
    service = IndexingService(
        store=store,
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=200,
        chunk_overlap=40,
    )
    return service, store


def test_index_document_persists_chunks_and_matching_vectors(
    tmp_path, sample_text_file, fake_cohere_client, fake_pinecone_client
):
    service, store = _service(tmp_path, fake_cohere_client, fake_pinecone_client)

    result = service.index_document(sample_text_file)

    assert result.chunk_count > 0
    assert store.count() == result.chunk_count
    stored_ids = {chunk.chunk_id for chunk in store.all_chunks()}
    assert set(fake_pinecone_client.index.vectors) == stored_ids


def test_index_document_embeds_with_document_input_type(
    tmp_path, sample_text_file, fake_cohere_client, fake_pinecone_client
):
    service, _ = _service(tmp_path, fake_cohere_client, fake_pinecone_client)
    service.index_document(sample_text_file)

    assert fake_cohere_client.calls
    assert all(
        call["input_type"] == "search_document" for call in fake_cohere_client.calls
    )


def test_rebuild_bm25_covers_all_stored_chunks(
    tmp_path, sample_text_file, fake_cohere_client, fake_pinecone_client
):
    service, store = _service(tmp_path, fake_cohere_client, fake_pinecone_client)
    service.index_document(sample_text_file)

    index = service.rebuild_bm25()

    assert len(index) == store.count()


def test_reindexing_same_file_does_not_duplicate(
    tmp_path, sample_text_file, fake_cohere_client, fake_pinecone_client
):
    service, store = _service(tmp_path, fake_cohere_client, fake_pinecone_client)

    first = service.index_document(sample_text_file)
    count_after_first = store.count()
    second = service.index_document(sample_text_file)

    assert second.document_id == first.document_id
    assert store.count() == count_after_first
