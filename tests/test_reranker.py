"""CohereReranker behavior, verified against a fake client (no network)."""

from __future__ import annotations

import pytest

from src.models.chunk import Chunk
from src.reranking.reranker import CohereReranker, RerankError
from tests.conftest import FakeCohereClient


def _chunk(chunk_id: str, position: int) -> Chunk:
    return Chunk(
        document_id="doc",
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        source_title="src.txt",
        position=position,
    )


def _candidates(n: int) -> list[Chunk]:
    return [_chunk(f"c{i}", i) for i in range(n)]


def test_sends_chunk_texts_in_candidate_order(fake_cohere_client):
    reranker = CohereReranker(client=fake_cohere_client)
    candidates = _candidates(3)
    reranker.rerank("q", candidates, top_n=3)
    assert fake_cohere_client.rerank_calls[0]["documents"] == [
        "text c0",
        "text c1",
        "text c2",
    ]


def test_maps_results_back_to_source_chunks_by_index(fake_cohere_client):
    reranker = CohereReranker(client=fake_cohere_client)
    candidates = _candidates(3)
    results = reranker.rerank("q", candidates, top_n=3)
    # The fake ranks the last document first, so c2 leads and c0 trails.
    assert [r.chunk.chunk_id for r in results] == ["c2", "c1", "c0"]


def test_results_are_ordered_by_descending_relevance(fake_cohere_client):
    reranker = CohereReranker(client=fake_cohere_client)
    scores = [r.relevance_score for r in reranker.rerank("q", _candidates(3), top_n=3)]
    assert scores == sorted(scores, reverse=True)


def test_top_n_is_capped_to_candidate_count(fake_cohere_client):
    reranker = CohereReranker(client=fake_cohere_client)
    reranker.rerank("q", _candidates(2), top_n=10)
    assert fake_cohere_client.rerank_calls[0]["top_n"] == 2


def test_max_tokens_per_doc_is_passed_through(fake_cohere_client):
    reranker = CohereReranker(client=fake_cohere_client, max_tokens_per_doc=1024)
    reranker.rerank("q", _candidates(1), top_n=1)
    assert fake_cohere_client.rerank_calls[0]["max_tokens_per_doc"] == 1024


def test_empty_candidates_makes_no_call(fake_cohere_client):
    reranker = CohereReranker(client=fake_cohere_client)
    assert reranker.rerank("q", [], top_n=5) == []
    assert fake_cohere_client.rerank_calls == []


def test_top_n_zero_makes_no_call(fake_cohere_client):
    reranker = CohereReranker(client=fake_cohere_client)
    assert reranker.rerank("q", _candidates(3), top_n=0) == []
    assert fake_cohere_client.rerank_calls == []


def test_provider_failure_raises_safe_rerank_error_preserving_cause():
    boom = ValueError("cohere internal detail with request id 12345")
    client = FakeCohereClient(rerank_error=boom)
    reranker = CohereReranker(client=client)

    with pytest.raises(RerankError) as excinfo:
        reranker.rerank("q", _candidates(3), top_n=3)

    # Client-safe message: no provider internals leak into the raised error.
    assert "internal detail" not in str(excinfo.value)
    assert "12345" not in str(excinfo.value)
    # Original error is preserved for server-side logging only.
    assert excinfo.value.__cause__ is boom
