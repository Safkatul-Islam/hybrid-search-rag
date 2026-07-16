"""DenseRetriever behavior, verified against fakes (no network)."""

from __future__ import annotations

from src.embeddings.embedder import CohereEmbedder
from src.models.chunk import Chunk
from src.retrieval.dense import DenseRetriever
from src.storage.vector_store import PineconeVectorStore


def _chunk(chunk_id: str, position: int) -> Chunk:
    return Chunk(
        document_id="doc",
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        source_title="src.txt",
        position=position,
    )


def _retriever(cohere, pinecone) -> tuple[DenseRetriever, PineconeVectorStore]:
    embedder = CohereEmbedder(client=cohere, dimension=4)
    vector_store = PineconeVectorStore(client=pinecone, dimension=4)
    return DenseRetriever(embedder=embedder, vector_store=vector_store), vector_store


def test_retrieve_embeds_query_with_search_query_input_type(
    fake_cohere_client, fake_pinecone_client
):
    retriever, store = _retriever(fake_cohere_client, fake_pinecone_client)
    store.upsert([_chunk("x", 0)], [[1.0, 0.0, 0.0, 0.0]])

    retriever.retrieve("hello", top_n=3)

    assert fake_cohere_client.calls[-1]["input_type"] == "search_query"


def test_retrieve_returns_ranked_chunk_id_score_pairs(
    fake_cohere_client, fake_pinecone_client
):
    retriever, store = _retriever(fake_cohere_client, fake_pinecone_client)
    store.upsert(
        [_chunk("x", 0), _chunk("y", 1)],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    )

    results = retriever.retrieve("hello", top_n=2)

    assert results[0][0] == "x"
    assert {chunk_id for chunk_id, _ in results} == {"x", "y"}


def test_retrieve_top_n_zero_short_circuits_without_embedding(
    fake_cohere_client, fake_pinecone_client
):
    retriever, _ = _retriever(fake_cohere_client, fake_pinecone_client)
    assert retriever.retrieve("hello", top_n=0) == []
    assert fake_cohere_client.calls == []
