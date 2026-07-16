"""End-to-end query pipeline, wired with real logic and fake provider clients.

Real ChunkStore + BM25Index + HybridRetriever + reranker + RagGenerator; only
the three provider clients (Cohere, Pinecone, Anthropic) are fakes. No network.
"""

from __future__ import annotations

from src.embeddings.embedder import CohereEmbedder
from src.generation.generator import RagGenerator
from src.llm.client import ClaudeClient
from src.models.chunk import Chunk
from src.prompts.templates import FALLBACK_ANSWER
from src.reranking.reranker import CohereReranker
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.services.query_service import QueryService
from src.storage.chunk_store import ChunkStore
from src.storage.vector_store import PineconeVectorStore
from tests.conftest import FakeAnthropicClient, FakeCohereClient, FakePineconeClient


def _chunks() -> list[Chunk]:
    texts = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "eta theta iota",
        "kappa lambda mu",
    ]
    return [
        Chunk(
            document_id="doc",
            chunk_id=f"doc-{i:04d}",
            text=text,
            source_title="s.txt",
            position=i,
            page=i + 1,
        )
        for i, text in enumerate(texts)
    ]


def _build(
    tmp_path,
    *,
    chunks: list[Chunk],
    cohere: FakeCohereClient,
    pinecone: FakePineconeClient,
    anthropic: FakeAnthropicClient,
) -> QueryService:
    store = ChunkStore(tmp_path / "chunks.sqlite")
    vector_store = PineconeVectorStore(client=pinecone, dimension=8)
    if chunks:
        store.upsert_chunks(chunks)
        vectors = [[float(i + 1)] + [0.0] * 7 for i in range(len(chunks))]
        vector_store.upsert(chunks, vectors)

    dense = DenseRetriever(
        embedder=CohereEmbedder(client=cohere, dimension=8),
        vector_store=vector_store,
    )
    hybrid = HybridRetriever(
        dense=dense,
        bm25=BM25Index(chunks),
        dense_top_n=10,
        bm25_top_n=10,
        rrf_k=60,
        fusion_top_n=10,
    )
    return QueryService(
        hybrid=hybrid,
        store=store,
        reranker=CohereReranker(client=cohere),
        generator=RagGenerator(llm=ClaudeClient(client=anthropic)),
        rerank_top_n=5,
    )


def test_happy_path_returns_grounded_cited_answer(tmp_path, fake_pinecone_client):
    cohere = FakeCohereClient()
    anthropic = FakeAnthropicClient(text="Grounded answer [1].")
    service = _build(
        tmp_path,
        chunks=_chunks(),
        cohere=cohere,
        pinecone=fake_pinecone_client,
        anthropic=anthropic,
    )

    result = service.answer("what is alpha")

    assert result.is_fallback is False
    assert result.rerank_failed is False
    assert result.answer == "Grounded answer [1]."
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id in result.used_chunk_ids
    assert result.used_chunk_ids  # non-empty


def test_empty_query_short_circuits_without_provider_calls(
    tmp_path, fake_pinecone_client
):
    cohere = FakeCohereClient()
    anthropic = FakeAnthropicClient()
    service = _build(
        tmp_path,
        chunks=_chunks(),
        cohere=cohere,
        pinecone=fake_pinecone_client,
        anthropic=anthropic,
    )

    result = service.answer("   ")

    assert result.is_fallback is True
    assert result.answer == FALLBACK_ANSWER
    assert cohere.calls == []
    assert cohere.rerank_calls == []
    assert anthropic.create_calls == []


def test_no_candidates_falls_back(tmp_path, fake_pinecone_client):
    service = _build(
        tmp_path,
        chunks=[],
        cohere=FakeCohereClient(),
        pinecone=fake_pinecone_client,
        anthropic=FakeAnthropicClient(text="unused"),
    )

    result = service.answer("anything")

    assert result.is_fallback is True
    assert result.answer == FALLBACK_ANSWER
    assert result.used_chunk_ids == []


def test_rerank_failure_degrades_and_is_flagged(tmp_path, fake_pinecone_client):
    # rerank raises, embedding still works -> degrade to fusion order, flagged.
    cohere = FakeCohereClient(rerank_error=RuntimeError("rerank boom"))
    anthropic = FakeAnthropicClient(text="Degraded answer [1].")
    service = _build(
        tmp_path,
        chunks=_chunks(),
        cohere=cohere,
        pinecone=fake_pinecone_client,
        anthropic=anthropic,
    )

    result = service.answer("what is alpha")

    assert result.rerank_failed is True
    assert result.is_fallback is False
    assert result.used_chunk_ids  # answered from fusion order
    assert len(result.citations) == 1


def test_hallucinated_citation_surfaces_in_result(tmp_path, fake_pinecone_client):
    cohere = FakeCohereClient()
    anthropic = FakeAnthropicClient(text="Real [1] and fake [99].")
    service = _build(
        tmp_path,
        chunks=_chunks(),
        cohere=cohere,
        pinecone=fake_pinecone_client,
        anthropic=anthropic,
    )

    result = service.answer("what is alpha")

    assert 99 in result.invalid_citation_numbers
    assert [c.number for c in result.citations] == [1]
