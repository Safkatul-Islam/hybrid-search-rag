"""BM25Index behavior — runs entirely locally, no providers involved."""

from __future__ import annotations

from src.models.chunk import Chunk
from src.retrieval.bm25 import BM25Index, tokenize


def _chunk(chunk_id: str, text: str, position: int) -> Chunk:
    return Chunk(
        document_id="doc",
        chunk_id=chunk_id,
        text=text,
        source_title="src.txt",
        position=position,
    )


def test_tokenize_lowercases_and_splits_on_word_boundaries():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_query_ranks_the_relevant_chunk_first():
    chunks = [
        _chunk("c0", "the cat sat on the mat", 0),
        _chunk("c1", "reciprocal rank fusion combines rankings", 1),
        _chunk("c2", "dogs and cats are common pets", 2),
    ]
    results = BM25Index(chunks).query("rank fusion", top_n=3)
    assert results[0][0] == "c1"


def test_query_respects_top_n():
    chunks = [_chunk(f"c{i}", f"term{i} shared", i) for i in range(5)]
    results = BM25Index(chunks).query("shared", top_n=2)
    assert len(results) == 2


def test_results_are_keyed_by_chunk_id():
    # A rare term needs a larger corpus to earn a positive BM25 IDF: a term that
    # occurs in half a two-doc corpus has IDF ~0 and carries no signal.
    chunks = [
        _chunk("a", "alpha beta gamma", 0),
        _chunk("b", "unicorn sparkle rainbow", 1),
        _chunk("c", "common ordinary words", 2),
        _chunk("d", "more ordinary text", 3),
    ]
    results = BM25Index(chunks).query("unicorn", top_n=1)
    assert results[0][0] == "b"


def test_empty_index_returns_empty():
    index = BM25Index([])
    assert len(index) == 0
    assert index.query("anything", top_n=5) == []


def test_top_n_zero_returns_empty():
    chunks = [_chunk("a", "alpha", 0)]
    assert BM25Index(chunks).query("alpha", top_n=0) == []
