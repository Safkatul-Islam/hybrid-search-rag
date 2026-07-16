"""Deterministic identity is the foundation of the whole pipeline."""

from __future__ import annotations

from src.models.chunk import compute_document_id, make_chunk_id


def test_document_id_is_deterministic():
    content = b"the same bytes produce the same id"
    assert compute_document_id(content) == compute_document_id(content)


def test_document_id_differs_for_different_content():
    assert compute_document_id(b"alpha") != compute_document_id(b"beta")


def test_chunk_id_is_deterministic():
    document_id = compute_document_id(b"doc")
    assert make_chunk_id(document_id, 0) == make_chunk_id(document_id, 0)


def test_chunk_id_is_positional_and_prefixed():
    document_id = compute_document_id(b"doc")
    assert make_chunk_id(document_id, 0) != make_chunk_id(document_id, 1)
    assert make_chunk_id(document_id, 3).startswith(document_id)
