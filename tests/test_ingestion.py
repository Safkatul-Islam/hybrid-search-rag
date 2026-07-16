"""Loading and chunking of text and PDF documents."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.loader import load


def test_load_text_file(sample_text_file):
    doc = load(sample_text_file)
    assert doc.source_title == "sample.txt"
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number is None
    assert "Retrieval augmented generation" in doc.pages[0].text


def test_load_pdf_file(sample_pdf_file):
    doc = load(sample_pdf_file)
    assert doc.source_title == "sample.pdf"
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert "Hello" in doc.pages[0].text


def test_unsupported_file_type_raises(tmp_path):
    bad = tmp_path / "data.csv"
    bad.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError):
        load(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "does_not_exist.txt")


def test_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        load(empty)


def test_chunking_is_stable_across_reingest(sample_text_file):
    first = chunk_document(load(sample_text_file), chunk_size=200, chunk_overlap=40)
    second = chunk_document(load(sample_text_file), chunk_size=200, chunk_overlap=40)

    assert len(first) > 1
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_chunk_positions_are_sequential_and_share_document_id(sample_text_file):
    chunks = chunk_document(load(sample_text_file), chunk_size=200, chunk_overlap=40)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert len({c.document_id for c in chunks}) == 1
