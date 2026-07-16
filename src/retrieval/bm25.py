"""Local BM25 keyword index built from the canonical chunk store.

BM25 keeps no persistent state: it is rebuilt from SQLite (the source of truth)
on startup. The same tokenizer is used for both indexing and querying, which is
an invariant — mismatched preprocessing would make scores meaningless. Results
are keyed by ``chunk_id`` so they align with the dense index.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from src.models.chunk import Chunk

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercased word-token tokenizer used for indexing and querying alike."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """An in-memory BM25 index over a set of chunks."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        corpus = [tokenize(chunk.text) for chunk in chunks]
        # BM25Okapi requires a non-empty corpus.
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def __len__(self) -> int:
        return len(self._chunk_ids)

    def query(self, text: str, top_n: int) -> list[tuple[str, float]]:
        """Return up to ``top_n`` ``(chunk_id, score)`` pairs, highest score first."""
        if self._bm25 is None or top_n < 1:
            return []
        scores = self._bm25.get_scores(tokenize(text))
        scored = [
            (chunk_id, float(score))
            for chunk_id, score in zip(self._chunk_ids, scores, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]
