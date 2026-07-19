"""Parse and validate numbered citations against the context that was shown.

The generator numbers the context chunks ``1..N``; the model cites by those
numbers. Here we extract the cited numbers from the answer and resolve them
against that same numbering. A citation can therefore only ever resolve to a
chunk that was actually in the context — the guard against fabricated sources.
Numbers outside ``1..N`` are reported as invalid rather than trusted or hidden.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.models.chunk import Chunk

# Matches a bracketed group of one or more numbers, e.g. [1], [2][3], [1, 2].
_BRACKET_RE = re.compile(r"\[([\d,\s]+)\]")
_NUMBER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class Citation:
    """A validated citation: the display number and the chunk it resolves to."""

    number: int
    chunk_id: str
    source_title: str
    page: int | None


def parse_citations(answer: str) -> list[int]:
    """Return the distinct citation numbers referenced in ``answer``, in order."""
    seen: dict[int, None] = {}
    for bracket in _BRACKET_RE.findall(answer):
        for token in _NUMBER_RE.findall(bracket):
            seen.setdefault(int(token), None)
    return list(seen)


def validate_and_resolve(
    cited: Sequence[int],
    numbered_chunks: Sequence[tuple[int, Chunk]],
) -> tuple[list[Citation], list[int]]:
    """Split cited numbers into resolved citations and invalid (out-of-range) ones.

    Returns:
        ``(citations, invalid_numbers)`` — ``citations`` for numbers that map to
        a context chunk, ``invalid_numbers`` for numbers with no such chunk.
    """
    by_number = {number: chunk for number, chunk in numbered_chunks}
    citations: list[Citation] = []
    invalid: list[int] = []
    for number in cited:
        chunk = by_number.get(number)
        if chunk is None:
            invalid.append(number)
            continue
        citations.append(
            Citation(
                number=number,
                chunk_id=chunk.chunk_id,
                source_title=chunk.source_title,
                page=chunk.page,
            )
        )
    return citations, invalid
