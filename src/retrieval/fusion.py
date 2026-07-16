"""Reciprocal rank fusion (RRF).

Combines several ranked lists of ``chunk_id`` into one. RRF uses only each item's
*rank position*, never its raw score: dense cosine scores and BM25 scores live on
different, incomparable scales, so averaging them would be meaningless. Each list
contributes ``1 / (k + rank)`` (rank starting at 1); an item's fused score is the
sum of its contributions across the lists it appears in.

Results are deduplicated by ``chunk_id`` and ordered by fused score descending,
with ties broken by ``chunk_id`` so the output is deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ``chunk_id`` lists into one ranking.

    Args:
        rankings: ranked lists of ``chunk_id`` (each already ordered best-first,
            and distinct within itself).
        k: RRF constant; larger values flatten the advantage of top ranks.
        top_n: if given, truncate the fused ranking to this many results.

    Returns:
        ``(chunk_id, fused_score)`` pairs, highest score first.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if top_n is not None:
        if top_n < 1:
            return []
        fused = fused[:top_n]
    return fused
