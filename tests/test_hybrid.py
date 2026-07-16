"""HybridRetriever — dense + BM25 fused, verified against fakes (no network)."""

from __future__ import annotations

from src.embeddings.embedder import CohereEmbedder
from src.models.chunk import Chunk
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.storage.vector_store import PineconeVectorStore


def _chunk(chunk_id: str, text: str, position: int) -> Chunk:
    return Chunk(
        document_id="doc",
        chunk_id=chunk_id,
        text=text,
        source_title="src.txt",
        position=position,
    )


def _hybrid(fake_cohere_client, fake_pinecone_client) -> HybridRetriever:
    chunks = [
        _chunk("c0", "reciprocal rank fusion combines rankings", 0),
        _chunk("c1", "the cat sat on the mat", 1),
        _chunk("c2", "dense embeddings and vectors", 2),
    ]
    # Vectors (dim 2): the query embeds to an all-equal vector, so a larger
    # component sum means a higher dense score. c0 has the largest sum.
    vectors = [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]

    embedder = CohereEmbedder(client=fake_cohere_client, dimension=2)
    vector_store = PineconeVectorStore(client=fake_pinecone_client, dimension=2)
    vector_store.upsert(chunks, vectors)

    dense = DenseRetriever(embedder=embedder, vector_store=vector_store)
    return HybridRetriever(
        dense=dense,
        bm25=BM25Index(chunks),
        dense_top_n=10,
        bm25_top_n=10,
        rrf_k=60,
        fusion_top_n=10,
    )


def test_chunk_favored_by_both_retrievers_ranks_first(
    fake_cohere_client, fake_pinecone_client
):
    hybrid = _hybrid(fake_cohere_client, fake_pinecone_client)
    results = hybrid.retrieve("rank fusion")
    assert results[0][0] == "c0"


def test_all_retrieved_chunks_appear_once(fake_cohere_client, fake_pinecone_client):
    hybrid = _hybrid(fake_cohere_client, fake_pinecone_client)
    ids = [chunk_id for chunk_id, _ in hybrid.retrieve("rank fusion")]
    assert sorted(ids) == ["c0", "c1", "c2"]
    assert len(ids) == len(set(ids))


def test_fusion_top_n_truncates_final_ranking(
    fake_cohere_client, fake_pinecone_client
):
    chunks = [_chunk(f"c{i}", f"term{i} shared text", i) for i in range(5)]
    vectors = [[float(i), 0.0] for i in range(5)]
    embedder = CohereEmbedder(client=fake_cohere_client, dimension=2)
    vector_store = PineconeVectorStore(client=fake_pinecone_client, dimension=2)
    vector_store.upsert(chunks, vectors)
    hybrid = HybridRetriever(
        dense=DenseRetriever(embedder=embedder, vector_store=vector_store),
        bm25=BM25Index(chunks),
        dense_top_n=10,
        bm25_top_n=10,
        rrf_k=60,
        fusion_top_n=2,
    )
    assert len(hybrid.retrieve("shared")) == 2
