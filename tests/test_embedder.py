"""CohereEmbedder behavior, verified against a fake client (no network)."""

from __future__ import annotations

import pytest

from src.embeddings.embedder import CohereEmbedder


def test_embed_documents_uses_search_document_input_type(fake_cohere_client):
    embedder = CohereEmbedder(client=fake_cohere_client, dimension=1024)
    embedder.embed_documents(["a", "bb"])
    assert fake_cohere_client.calls[0]["input_type"] == "search_document"


def test_embed_query_uses_search_query_input_type(fake_cohere_client):
    embedder = CohereEmbedder(client=fake_cohere_client, dimension=1024)
    vector = embedder.embed_query("hello")
    assert fake_cohere_client.calls[0]["input_type"] == "search_query"
    assert len(vector) == 1024


def test_embed_documents_returns_one_vector_per_text_at_configured_dimension(
    fake_cohere_client,
):
    embedder = CohereEmbedder(client=fake_cohere_client, dimension=1024)
    vectors = embedder.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 1024 for v in vectors)


def test_embed_documents_batches_over_the_96_text_limit(fake_cohere_client):
    embedder = CohereEmbedder(client=fake_cohere_client, dimension=8, batch_size=96)
    vectors = embedder.embed_documents([f"t{i}" for i in range(200)])
    assert len(vectors) == 200
    assert [len(call["texts"]) for call in fake_cohere_client.calls] == [96, 96, 8]


def test_output_dimension_is_passed_through(fake_cohere_client):
    embedder = CohereEmbedder(client=fake_cohere_client, dimension=1024)
    embedder.embed_documents(["x"])
    assert fake_cohere_client.calls[0]["output_dimension"] == 1024


def test_empty_documents_makes_no_call(fake_cohere_client):
    embedder = CohereEmbedder(client=fake_cohere_client, dimension=1024)
    assert embedder.embed_documents([]) == []
    assert fake_cohere_client.calls == []


def test_invalid_batch_size_raises(fake_cohere_client):
    with pytest.raises(ValueError):
        CohereEmbedder(client=fake_cohere_client, batch_size=0)
    with pytest.raises(ValueError):
        CohereEmbedder(client=fake_cohere_client, batch_size=97)
