"""Reciprocal rank fusion — pure logic, no providers involved."""

from __future__ import annotations

import pytest

from src.retrieval.fusion import reciprocal_rank_fusion


def _ids(fused: list[tuple[str, float]]) -> list[str]:
    return [chunk_id for chunk_id, _ in fused]


def test_item_ranked_well_in_both_lists_wins():
    dense = ["a", "b", "c"]
    bm25 = ["b", "a", "d"]
    # "b" is 2nd + 1st; "a" is 1st + 2nd. With equal rank sums the score ties,
    # so add a third list that favors "b" to make the win unambiguous.
    result = reciprocal_rank_fusion([dense, bm25, ["b"]])
    assert _ids(result)[0] == "b"


def test_item_in_only_one_list_is_still_included():
    result = reciprocal_rank_fusion([["a", "b"], ["c"]])
    assert set(_ids(result)) == {"a", "b", "c"}


def test_duplicate_ids_across_lists_are_merged():
    result = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    assert _ids(result).count("a") == 1


def test_higher_k_flattens_the_top_rank_advantage():
    lists = [["a", "b"], ["a", "b"]]
    low_k = dict(reciprocal_rank_fusion(lists, k=1))
    high_k = dict(reciprocal_rank_fusion(lists, k=1000))
    # The gap between the 1st and 2nd item shrinks as k grows.
    assert (low_k["a"] - low_k["b"]) > (high_k["a"] - high_k["b"])


def test_ties_break_deterministically_by_chunk_id():
    # Symmetric input: "a" and "b" earn identical scores; order must be stable.
    result = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert _ids(result) == ["a", "b"]


def test_top_n_truncates_the_fused_ranking():
    result = reciprocal_rank_fusion([["a", "b", "c", "d"]], top_n=2)
    assert len(result) == 2


def test_top_n_zero_returns_empty():
    assert reciprocal_rank_fusion([["a", "b"]], top_n=0) == []


def test_empty_rankings_return_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
