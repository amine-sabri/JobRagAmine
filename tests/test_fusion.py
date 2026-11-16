"""RRF, checked against hand-computed scores."""

import pytest

from jobrag.retrieve.fusion import fuse_hits, reciprocal_rank_fusion


def test_a_document_first_in_both_lists_scores_twice():
    # 1/(60+1) + 1/(60+1) = 2/61 = 0.0327868…
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    assert fused[0][0] == "a"
    assert fused[0][1] == pytest.approx(2 / 61)


def test_agreement_beats_a_single_first_place():
    # The property RRF exists for:
    #   "b" is 2nd in both lists -> 1/62 + 1/62 = 0.032258
    #   "a" is 1st in one only   -> 1/61        = 0.016393
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["c", "b"]]))
    assert fused["b"] == pytest.approx(2 / 62)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] > fused["a"]


def test_ranking_order_is_by_score_then_id():
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
    # Both appear once at rank 1 and once at rank 2: identical scores, so the
    # id decides and the result is stable across runs.
    assert [identifier for identifier, _ in fused] == ["x", "y"]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_k_flattens_the_weighting():
    small_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    large_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))
    # With k=1: 1/2 vs 1/3 — a 1.5x gap. With k=1000: 1/1001 vs 1/1002 — none.
    assert small_k["a"] / small_k["b"] == pytest.approx(1.5)
    assert large_k["a"] / large_k["b"] == pytest.approx(1.001, abs=1e-3)


def test_weights_shift_the_balance():
    # Vector list first, keyword list second, keyword trusted twice as much.
    fused = dict(reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 2.0]))
    assert fused["b"] == pytest.approx(2 / 61)
    assert fused["a"] == pytest.approx(1 / 61)


def test_one_empty_ranking_is_not_an_error():
    fused = reciprocal_rank_fusion([["a", "b"], []])
    assert [identifier for identifier, _ in fused] == ["a", "b"]
    assert reciprocal_rank_fusion([[], []]) == []


def test_a_duplicate_inside_one_ranking_counts_at_its_best_rank_only_once_per_list():
    # Defensive: a retriever returning the same id twice must not double-score it
    # beyond its two ranks, and must not crash the fusion.
    fused = dict(reciprocal_rank_fusion([["a", "a"]]))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)


def test_weights_must_match_the_rankings():
    with pytest.raises(ValueError, match="one weight per ranking"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([["a"]], k=0)


def test_fuse_hits_keeps_the_payload_and_explains_the_rank():
    vector = [{"chunk_id": "c1", "text": "A", "score": 0.81}, {"chunk_id": "c2", "text": "B"}]
    keyword = [{"chunk_id": "c2", "text": "B"}, {"chunk_id": "c3", "text": "C"}]
    fused = fuse_hits([vector, keyword])
    top = fused[0]
    assert top["chunk_id"] == "c2"  # 2nd and 1st beats 1st and absent
    assert top["text"] == "B"
    assert top["ranks"] == [2, 1]
    assert top["rrf_score"] == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)
    # A chunk only the vector side found keeps a None for the keyword rank.
    c1 = next(row for row in fused if row["chunk_id"] == "c1")
    assert c1["ranks"] == [1, None]
    assert c1["score"] == 0.81


def test_fuse_hits_respects_the_limit():
    vector = [{"chunk_id": f"c{i}"} for i in range(10)]
    assert len(fuse_hits([vector], limit=3)) == 3
