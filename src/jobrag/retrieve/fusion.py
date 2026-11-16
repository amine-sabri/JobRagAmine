"""Reciprocal Rank Fusion: combining a vector ranking with a keyword ranking.

RRF scores a document by its *rank* in each list, never by the scores the
retrievers produced:

    score(d) = Σ  weight_i / (k + rank_i(d))         rank starting at 1

That is the whole point. A cosine similarity of 0.82 and a BM25 score of 14.3
are not comparable, and normalising them means inventing a calibration neither
retriever offers. Ranks are comparable by construction.

The constant k (60 by convention, from Cormack et al. 2009) damps the top of
each list: without it, rank 1 would be worth twice rank 2, and a single
retriever's favourite would win every fusion. Larger k flattens the weighting.
"""

from collections.abc import Sequence
from typing import Any


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one, best first.

    Ties are broken by id so the output is deterministic: two runs of the same
    evaluation must not differ because a dict happened to iterate differently.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is required")
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def fuse_hits(
    runs: Sequence[Sequence[dict[str, Any]]],
    *,
    key: str = "chunk_id",
    k: int = 60,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse result rows, keeping each row's payload and where it came from.

    The returned rows carry `rrf_score` and `ranks` (the 1-based rank in each
    input run, None when absent), because "why is this chunk first?" is the
    question a failure case always raises.
    """
    rankings = [[row[key] for row in run] for run in runs]
    fused = reciprocal_rank_fusion(rankings, k=k, weights=weights)

    payloads: dict[str, dict[str, Any]] = {}
    ranks: dict[str, list[int | None]] = {}
    for run_index, run in enumerate(runs):
        for rank, row in enumerate(run, start=1):
            identifier = row[key]
            payloads.setdefault(identifier, dict(row))
            ranks.setdefault(identifier, [None] * len(runs))
            if ranks[identifier][run_index] is None:
                ranks[identifier][run_index] = rank

    out = []
    for identifier, score in fused[: limit or len(fused)]:
        row = dict(payloads[identifier])
        row["rrf_score"] = round(score, 6)
        row["ranks"] = ranks[identifier]
        out.append(row)
    return out
