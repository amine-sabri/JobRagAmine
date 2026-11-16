"""Retrieval metrics.

Two decisions shape this module.

**Relevance is labelled per offer, retrieval returns chunks.** An offer split
into four chunks could otherwise fill the top-4 and look like four successes.
Every metric therefore runs on the offer ids behind the ranked chunks, keeping
each offer at its best (first) position.

**Questions with no answer are not scored here.** A "none" question has no
relevant offer, so recall and MRR are undefined for it — averaging a 0 or a 1
into them would quietly move the headline number. They are excluded, counted,
and measured separately by the refusal rate in phase 7.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any


def offers_from_chunks(chunk_hits: Sequence[dict[str, Any]]) -> list[str]:
    """Ranked offer ids behind ranked chunk hits, de-duplicated, order kept.

    >>> offers_from_chunks([{"offer_id": "a"}, {"offer_id": "a"}, {"offer_id": "b"}])
    ['a', 'b']
    """
    seen: dict[str, None] = {}
    for hit in chunk_hits:
        seen.setdefault(hit["offer_id"], None)
    return list(seen)


def recall_at_k(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    """Share of the relevant offers that appear in the top k."""
    relevant = set(relevant)
    if not relevant:
        raise ValueError("recall is undefined with no relevant offer")
    found = len(relevant & set(ranked[:k]))
    return found / len(relevant)


def precision_at_k(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    """Share of the top k that is relevant. Divided by k, not by the hits."""
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(relevant) & set(ranked[:k])) / k


def hit_rate_at_k(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    """1.0 if at least one relevant offer is in the top k. Also called success@k."""
    return 1.0 if set(relevant) & set(ranked[:k]) else 0.0


def reciprocal_rank(relevant: Iterable[str], ranked: Sequence[str], k: int | None = None) -> float:
    """1/rank of the first relevant offer, 0.0 if none appears (in the top k)."""
    relevant = set(relevant)
    for position, offer_id in enumerate(ranked if k is None else ranked[:k], start=1):
        if offer_id in relevant:
            return 1.0 / position
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain, log2 discount, rank starting at 1."""
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    """nDCG with binary relevance: rewards putting relevant offers higher."""
    relevant = set(relevant)
    if not relevant:
        raise ValueError("nDCG is undefined with no relevant offer")
    gains = [1.0 if offer_id in relevant else 0.0 for offer_id in ranked[:k]]
    ideal = [1.0] * min(len(relevant), k)
    best = dcg(ideal)
    return dcg(gains) / best if best else 0.0


def average_precision(relevant: Iterable[str], ranked: Sequence[str], k: int) -> float:
    """Mean of the precisions measured at each relevant hit in the top k."""
    relevant = set(relevant)
    if not relevant:
        raise ValueError("average precision is undefined with no relevant offer")
    hits = 0
    total = 0.0
    for position, offer_id in enumerate(ranked[:k], start=1):
        if offer_id in relevant:
            hits += 1
            total += hits / position
    return total / min(len(relevant), k)


def recall_ceiling_at_k(relevant: Iterable[str], k: int) -> float:
    """The best recall@k this question allows: min(k, |relevant|) / |relevant|.

    A question with 9 relevant offers cannot score above 5/9 = 0.56 at k=5, so
    a raw recall@5 of 0.19 means something very different from 0.19 out of 1.0.
    Reported next to the metric, with the ratio, so the number is readable.
    """
    relevant = set(relevant)
    if not relevant:
        raise ValueError("the ceiling is undefined with no relevant offer")
    return min(k, len(relevant)) / len(relevant)


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """p50/p95/p99 and the mean, for latencies. Empty input gives an empty dict."""
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]

    return {
        "p50": round(at(0.50), 1),
        "p95": round(at(0.95), 1),
        "p99": round(at(0.99), 1),
        "mean": round(sum(ordered) / len(ordered), 1),
        "max": round(ordered[-1], 1),
    }


def evaluate_run(
    results: Sequence[tuple[Iterable[str], Sequence[str]]],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    latencies_ms: Sequence[float] = (),
) -> dict[str, Any]:
    """Aggregate one retrieval configuration over the evaluation set.

    `results` pairs the labelled offer ids with the ranked offer ids returned
    for that question. Questions with no labelled offer are skipped and counted:
    they belong to the refusal metric, not to recall.
    """
    scorable = [(set(relevant), ranked) for relevant, ranked in results if relevant]
    skipped = len(results) - len(scorable)
    if not scorable:
        return {
            "questions": 0,
            "skipped_no_answer": skipped,
            "latency_ms": percentiles(latencies_ms),
        }

    report: dict[str, Any] = {"questions": len(scorable), "skipped_no_answer": skipped}
    for k in ks:
        recall = sum(recall_at_k(r, ranked, k) for r, ranked in scorable) / len(scorable)
        ceiling = sum(recall_ceiling_at_k(r, k) for r, _ in scorable) / len(scorable)
        report[f"recall@{k}"] = round(recall, 4)
        report[f"recall_ceiling@{k}"] = round(ceiling, 4)
        # How much of the achievable recall was actually achieved.
        report[f"recall_of_ceiling@{k}"] = round(recall / ceiling, 4) if ceiling else 0.0
        report[f"hit@{k}"] = round(
            sum(hit_rate_at_k(r, ranked, k) for r, ranked in scorable) / len(scorable), 4
        )
        report[f"ndcg@{k}"] = round(
            sum(ndcg_at_k(r, ranked, k) for r, ranked in scorable) / len(scorable), 4
        )
    largest_k = max(ks)
    report[f"mrr@{largest_k}"] = round(
        sum(reciprocal_rank(r, ranked, largest_k) for r, ranked in scorable) / len(scorable), 4
    )
    report[f"map@{largest_k}"] = round(
        sum(average_precision(r, ranked, largest_k) for r, ranked in scorable) / len(scorable), 4
    )
    report["latency_ms"] = percentiles(latencies_ms)
    return report
