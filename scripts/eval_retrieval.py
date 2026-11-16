#!/usr/bin/env python
"""Measure retrieval against the hand-labelled evaluation set.

Runs each configuration over the same questions and reports recall@k, MRR,
nDCG and latency, so "the reranker helped" becomes a number instead of a
feeling.

Chunks are collapsed to their offers before scoring: relevance is labelled per
offer, and an offer split into four chunks must not count as four hits.
Questions with no answer are not scored here — they are counted and left to the
refusal rate of phase 7.

Usage:
    python scripts/eval_retrieval.py --modes keyword
    python scripts/eval_retrieval.py --modes vector,keyword,hybrid
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrag.evalset import load_questions  # noqa: E402
from jobrag.metrics import evaluate_run, offers_from_chunks  # noqa: E402
from jobrag.report import inject_into_file, markdown_table  # noqa: E402
from jobrag.retrieve.fusion import fuse_hits  # noqa: E402
from jobrag.retrieve.keyword import search_chunks as keyword_search  # noqa: E402
from jobrag.retrieve.vector import search_chunks as vector_search  # noqa: E402

MODES = ("vector", "keyword", "hybrid")


def run_mode(mode: str, question: str, *, depth: int, embedder, ef_search: int, rrf_k: int):
    """One question through one configuration. Returns (ranked offer ids, ms)."""
    started = time.perf_counter()
    if mode == "keyword":
        hits = keyword_search(question, k=depth)["results"]
    elif mode == "vector":
        hits = vector_search(question, embedder, k=depth, ef_search=ef_search)["results"]
    else:
        vector_hits = vector_search(question, embedder, k=depth, ef_search=ef_search)["results"]
        keyword_hits = keyword_search(question, k=depth)["results"]
        hits = fuse_hits([vector_hits, keyword_hits], k=rrf_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return offers_from_chunks(hits), elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="eval/questions.yaml", type=Path)
    parser.add_argument("--modes", default="vector,keyword,hybrid")
    parser.add_argument("--ks", default="1,3,5,10", help="cut-offs to report")
    parser.add_argument("--depth", type=int, default=20, help="hits retrieved before scoring")
    parser.add_argument("--ef-search", type=int, default=80)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--results", default="eval/results", type=Path)
    parser.add_argument("--readme", default="README.md", type=Path)
    parser.add_argument("--no-readme", action="store_true")
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = set(modes) - set(MODES)
    if unknown:
        print(f"unknown mode(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print(f"no question in {args.questions} — write some with `make label`", file=sys.stderr)
        return 2

    ks = [int(k) for k in args.ks.split(",")]
    embedder = None
    if {"vector", "hybrid"} & set(modes):
        from jobrag.embed import get_embedder

        embedder = get_embedder(threads=4)
        print(f"loading {embedder.model_name} …", flush=True)
        # Load the model before the timing loop, so the first question does not
        # carry a 50-second model load in its latency.
        dimensions = embedder.dimension
        print(f"  {dimensions} dimensions", flush=True)

    reports: dict[str, dict] = {}
    failures: dict[str, list[dict]] = {}
    for mode in modes:
        results = []
        latencies = []
        per_question = []
        for question in questions:
            relevant = question.get("relevant_offer_ids") or []
            ranked, elapsed_ms = run_mode(
                mode,
                question["question"],
                depth=args.depth,
                embedder=embedder,
                ef_search=args.ef_search,
                rrf_k=args.rrf_k,
            )
            results.append((relevant, ranked))
            latencies.append(elapsed_ms)
            per_question.append(
                {
                    "id": question["id"],
                    "type": question["type"],
                    "found_at": next(
                        (i + 1 for i, offer in enumerate(ranked) if offer in set(relevant)), None
                    ),
                    "ms": round(elapsed_ms, 1),
                }
            )
        reports[mode] = evaluate_run(results, ks=ks, latencies_ms=latencies)
        # The questions worth reading afterwards: labelled, and missed entirely.
        failures[mode] = [
            row for row in per_question if row["found_at"] is None and row["type"] != "none"
        ]
        print(f"{mode:<8} {reports[mode]}", flush=True)

    largest = max(ks)
    headers = [
        "configuration",
        *[f"recall@{k}" for k in ks],
        f"hit@{min(5, largest)}",
        f"mrr@{largest}",
        "p50 ms",
        "p95 ms",
    ]
    rows = []
    for mode in modes:
        report = reports[mode]
        rows.append(
            [
                mode,
                *[f"{report[f'recall@{k}']:.3f}" for k in ks],
                f"{report[f'hit@{min(5, largest)}']:.3f}",
                f"{report[f'mrr@{largest}']:.3f}",
                report["latency_ms"].get("p50", "-"),
                report["latency_ms"].get("p95", "-"),
            ]
        )
    ceilings = " · ".join(
        f"recall@{k} ≤ {reports[modes[0]][f'recall_ceiling@{k}']:.3f}" for k in ks
    )

    scored = reports[modes[0]]["questions"]
    skipped = reports[modes[0]]["skipped_no_answer"]
    command = f"python scripts/eval_retrieval.py --modes {','.join(modes)} --depth {args.depth}"
    md = f"""<!-- generated by: `{command}` -->
{scored} labelled questions scored, {skipped} "no answer" questions excluded (they are measured by
the refusal rate, not by recall). Retrieval depth {args.depth}, HNSW `ef_search={args.ef_search}`,
RRF `k={args.rrf_k}`. Latency is end to end per question, on CPU.

**Recall is capped by the labels**, because several questions have more relevant offers than there
are slots: {ceilings}. The `recall_of_ceiling@k` figures in `retrieval.json` give the share of the
achievable recall that was actually achieved.

{markdown_table(headers, rows)}
"""
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "retrieval.json").write_text(
        json.dumps(
            {"command": command, "reports": reports, "misses": failures},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    (args.results / "retrieval.md").write_text(md, encoding="utf-8")
    if not args.no_readme:
        inject_into_file(args.readme, "retrieval-results", md)
    print("\n" + md)
    for mode in modes:
        if failures[mode]:
            missed = ", ".join(row["id"] for row in failures[mode][:10])
            print(f"{mode}: missed {len(failures[mode])} questions entirely — {missed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
