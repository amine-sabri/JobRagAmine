"""Retrieval metrics, checked against values computed by hand.

Every expected number here is worked out in the comment next to it. A metric
that is only tested against its own implementation tests nothing.
"""

import math

import pytest

from jobrag.metrics import (
    average_precision,
    dcg,
    evaluate_run,
    hit_rate_at_k,
    ndcg_at_k,
    offers_from_chunks,
    percentiles,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_chunks_collapse_to_offers_keeping_the_best_rank():
    hits = [
        {"offer_id": "a", "chunk_id": "a:0"},
        {"offer_id": "a", "chunk_id": "a:3"},
        {"offer_id": "b", "chunk_id": "b:1"},
        {"offer_id": "a", "chunk_id": "a:7"},
    ]
    # One offer split into four chunks must not count as four successes.
    assert offers_from_chunks(hits) == ["a", "b"]


def test_offers_from_chunks_on_an_empty_result():
    assert offers_from_chunks([]) == []


@pytest.mark.parametrize(
    ("relevant", "ranked", "k", "expected"),
    [
        # 2 relevant, 1 of them in the top 5 -> 1/2
        ({"a", "b"}, ["x", "a", "y", "z", "w"], 5, 0.5),
        # both found -> 1.0
        ({"a", "b"}, ["a", "b", "x"], 5, 1.0),
        # relevant offer sits at rank 6, outside the top 5 -> 0.0
        ({"a"}, ["x", "y", "z", "w", "v", "a"], 5, 0.0),
        # 3 relevant, 2 within the top 3 -> 2/3
        ({"a", "b", "c"}, ["a", "b", "x"], 3, 2 / 3),
        # k larger than the result list is not an error
        ({"a"}, ["a"], 10, 1.0),
    ],
)
def test_recall_at_k(relevant, ranked, k, expected):
    assert recall_at_k(relevant, ranked, k) == pytest.approx(expected)


def test_recall_refuses_a_question_with_no_answer():
    # A "none" question must go to the refusal rate, never into recall.
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(set(), ["a", "b"], 5)


@pytest.mark.parametrize(
    ("relevant", "ranked", "k", "expected"),
    [
        ({"a", "b"}, ["a", "b", "x", "y", "z"], 5, 2 / 5),  # 2 relevant in 5 slots
        ({"a"}, ["a", "x", "y"], 3, 1 / 3),
        ({"a"}, ["x", "y", "z"], 3, 0.0),
    ],
)
def test_precision_at_k(relevant, ranked, k, expected):
    assert precision_at_k(relevant, ranked, k) == pytest.approx(expected)


def test_precision_rejects_k_zero():
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k({"a"}, ["a"], 0)


@pytest.mark.parametrize(
    ("ranked", "k", "expected"),
    [
        (["a", "x"], 5, 1.0),
        (["x", "y", "a"], 5, 1.0),
        (["x", "y", "a"], 2, 0.0),  # outside the window
        (["x", "y"], 5, 0.0),
    ],
)
def test_hit_rate(ranked, k, expected):
    assert hit_rate_at_k({"a"}, ranked, k) == expected


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [
        (["a", "x", "y"], 1.0),  # first position -> 1/1
        (["x", "a", "y"], 0.5),  # second -> 1/2
        (["x", "y", "a"], 1 / 3),  # third -> 1/3
        (["x", "y", "z"], 0.0),  # absent
    ],
)
def test_reciprocal_rank(ranked, expected):
    assert reciprocal_rank({"a"}, ranked) == pytest.approx(expected)


def test_reciprocal_rank_only_looks_at_the_top_k():
    assert reciprocal_rank({"a"}, ["x", "y", "z", "a"], k=3) == 0.0
    assert reciprocal_rank({"a"}, ["x", "y", "z", "a"], k=4) == pytest.approx(0.25)


def test_dcg_discounts_by_log2_of_the_rank():
    # gains 1, 0, 1 -> 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    assert dcg([1.0, 0.0, 1.0]) == pytest.approx(1.5)
    assert dcg([]) == 0.0


def test_ndcg_is_the_hand_computed_ratio():
    # relevant {a, b}; ranked a, x, b with k=3
    #   DCG  = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5          = 1.5
    #   IDCG = 1/log2(2) + 1/log2(3)     = 1 + 0.63092975…  = 1.63092975…
    #   nDCG = 1.5 / 1.63092975…         = 0.91972078…
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k({"a", "b"}, ["a", "x", "b"], 3) == pytest.approx(expected)
    assert round(expected, 4) == 0.9197


def test_ndcg_is_one_when_the_relevant_offers_are_on_top():
    assert ndcg_at_k({"a", "b"}, ["a", "b", "x"], 3) == pytest.approx(1.0)


def test_ndcg_rewards_a_higher_position():
    high = ndcg_at_k({"a"}, ["a", "x", "y"], 3)
    low = ndcg_at_k({"a"}, ["x", "y", "a"], 3)
    assert high > low
    assert low == pytest.approx(1 / 2)  # 1/log2(4) / 1 = 0.5


def test_average_precision_is_the_hand_computed_mean():
    # relevant {a, b}; ranked a, x, b, y, z with k=5
    #   hit at rank 1 -> precision 1/1
    #   hit at rank 3 -> precision 2/3
    #   AP = (1 + 0.6666…) / 2 = 0.8333…
    assert average_precision({"a", "b"}, ["a", "x", "b", "y", "z"], 5) == pytest.approx(5 / 6)


def test_average_precision_is_zero_without_a_hit():
    assert average_precision({"a"}, ["x", "y"], 5) == 0.0


def test_percentiles_of_latencies():
    values = [float(v) for v in range(1, 101)]  # 1..100
    result = percentiles(values)
    assert result["p50"] == 51.0
    assert result["p95"] == 96.0
    assert result["max"] == 100.0
    assert result["mean"] == 50.5
    assert percentiles([]) == {}


def test_evaluate_run_averages_and_skips_the_no_answer_questions():
    results = [
        ({"a"}, ["a", "x", "y", "z", "w"]),  # recall@5 = 1, rr = 1
        ({"b"}, ["x", "y", "b", "z", "w"]),  # recall@5 = 1, rr = 1/3
        ({"c"}, ["x", "y", "z", "w", "v"]),  # recall@5 = 0, rr = 0
        (set(), ["x", "y"]),  # a "none" question: skipped
    ]
    report = evaluate_run(results, ks=(1, 5), latencies_ms=[10.0, 20.0, 30.0])
    assert report["questions"] == 3
    assert report["skipped_no_answer"] == 1
    assert report["recall@5"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["recall@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["hit@5"] == pytest.approx(2 / 3, abs=1e-4)
    # MRR = (1 + 1/3 + 0) / 3 = 0.4444
    assert report["mrr@5"] == pytest.approx(4 / 9, abs=1e-4)
    assert report["latency_ms"]["p50"] == 20.0


def test_evaluate_run_on_an_evaluation_set_of_only_no_answer_questions():
    report = evaluate_run([(set(), ["x"]), (set(), [])], ks=(5,))
    assert report == {"questions": 0, "skipped_no_answer": 2, "latency_ms": {}}


def test_the_recall_ceiling_is_the_hand_computed_fraction():
    from jobrag.metrics import recall_ceiling_at_k

    # 9 relevant offers, 5 slots -> at best 5/9 of them can be found.
    assert recall_ceiling_at_k({f"o{i}" for i in range(9)}, 5) == pytest.approx(5 / 9)
    # Fewer relevant offers than slots: a perfect score is reachable.
    assert recall_ceiling_at_k({"a", "b"}, 5) == 1.0
    assert recall_ceiling_at_k({"a"}, 1) == 1.0
    with pytest.raises(ValueError, match="undefined"):
        recall_ceiling_at_k(set(), 5)


def test_evaluate_run_reports_recall_against_its_ceiling():
    # Two questions, 4 relevant offers each, one hit in the top 2 for each.
    # recall@2       = (1/4 + 1/4) / 2 = 0.25
    # ceiling@2      = min(2, 4)/4     = 0.5
    # recall/ceiling = 0.25 / 0.5      = 0.5
    results = [
        ({"a", "b", "c", "d"}, ["a", "x"]),
        ({"e", "f", "g", "h"}, ["e", "y"]),
    ]
    report = evaluate_run(results, ks=(2,))
    assert report["recall@2"] == pytest.approx(0.25)
    assert report["recall_ceiling@2"] == pytest.approx(0.5)
    assert report["recall_of_ceiling@2"] == pytest.approx(0.5)


def test_a_perfect_run_reaches_its_ceiling_exactly():
    results = [({"a", "b", "c"}, ["a", "b", "c", "x"])]
    report = evaluate_run(results, ks=(2, 5))
    assert report["recall_of_ceiling@2"] == pytest.approx(1.0)
    assert report["recall_of_ceiling@5"] == pytest.approx(1.0)
    assert report["recall@5"] == pytest.approx(1.0)
