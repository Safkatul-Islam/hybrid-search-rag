"""Round-trip and upsert behavior of the canonical SQLite chunk store."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.loader import load
from src.storage.chunk_store import ChunkStore


@pytest.fixture
def store(tmp_path):
    return ChunkStore(tmp_path / "chunks.sqlite")


def _chunks(sample_text_file):
    return chunk_document(load(sample_text_file), chunk_size=200, chunk_overlap=40)


def test_upsert_and_get_round_trip(store, sample_text_file):
    chunks = _chunks(sample_text_file)
    store.upsert_chunks(chunks)

    fetched = store.get_chunk(chunks[0].chunk_id)
    assert fetched == chunks[0]


def test_reingest_does_not_duplicate(store, sample_text_file):
    chunks = _chunks(sample_text_file)
    store.upsert_chunks(chunks)
    count_after_first = store.count()

    store.upsert_chunks(chunks)  # identical ids -> upsert, no growth
    assert store.count() == count_after_first
    assert count_after_first == len(chunks)


def test_get_chunks_by_document_is_ordered(store, sample_text_file):
    chunks = _chunks(sample_text_file)
    store.upsert_chunks(chunks)

    got = store.get_chunks_by_document(chunks[0].document_id)
    assert [c.chunk_id for c in got] == [c.chunk_id for c in chunks]
    assert [c.position for c in got] == list(range(len(chunks)))


def test_get_missing_chunk_returns_none(store):
    assert store.get_chunk("nonexistent") is None


def test_get_chunks_returns_in_requested_order(store, sample_text_file):
    chunks = _chunks(sample_text_file)
    store.upsert_chunks(chunks)

    ids = [chunks[2].chunk_id, chunks[0].chunk_id]
    got = store.get_chunks(ids)
    assert [c.chunk_id for c in got] == ids


def test_get_chunks_skips_missing_ids(store, sample_text_file):
    chunks = _chunks(sample_text_file)
    store.upsert_chunks(chunks)

    got = store.get_chunks([chunks[0].chunk_id, "nonexistent"])
    assert [c.chunk_id for c in got] == [chunks[0].chunk_id]


def test_get_chunks_empty_input_returns_empty(store):
    assert store.get_chunks([]) == []
