"""Citation parsing and validation — pure logic, no providers involved."""

from __future__ import annotations

from src.generation.citations import parse_citations, validate_and_resolve
from src.models.chunk import Chunk


def _chunk(chunk_id: str, position: int, page: int | None = None) -> Chunk:
    return Chunk(
        document_id="doc",
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        source_title="src.pdf",
        position=position,
        page=page,
    )


def _numbered(n: int) -> list[tuple[int, Chunk]]:
    return [(i, _chunk(f"c{i}", i, page=i)) for i in range(1, n + 1)]


def test_parses_single_and_adjacent_citations():
    assert parse_citations("Answer [1] and more [2][3].") == [1, 2, 3]


def test_parses_comma_separated_group():
    assert parse_citations("Both apply [1, 2].") == [1, 2]


def test_deduplicates_and_preserves_first_seen_order():
    assert parse_citations("[2] then [1] then [2] again") == [2, 1]


def test_no_citations_returns_empty():
    assert parse_citations("No brackets here.") == []


def test_valid_numbers_resolve_to_their_chunks():
    numbered = _numbered(3)
    citations, invalid = validate_and_resolve([1, 3], numbered)

    assert invalid == []
    assert [(c.number, c.chunk_id, c.page) for c in citations] == [
        (1, "c1", 1),
        (3, "c3", 3),
    ]


def test_out_of_range_numbers_are_reported_invalid_not_resolved():
    numbered = _numbered(2)
    citations, invalid = validate_and_resolve([1, 99], numbered)

    assert [c.chunk_id for c in citations] == ["c1"]
    assert invalid == [99]
