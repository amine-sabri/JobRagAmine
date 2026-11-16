#!/usr/bin/env python
"""Ask the assistant a question from the terminal.

Usage:
    python scripts/ask.py "Quelles offres demandent du Kubernetes à Lille ?"
    python scripts/ask.py --mode keyword --k 5 "offres en alternance ?"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrag.config import get_settings  # noqa: E402
from jobrag.generate import answer_question  # noqa: E402
from jobrag.llm import OllamaLLM  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+")
    parser.add_argument("--mode", default="hybrid", choices=("vector", "keyword", "hybrid"))
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--no-filters", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the whole payload")
    args = parser.parse_args()

    question = " ".join(args.question)
    settings = get_settings()
    llm = OllamaLLM(settings.ollama_model, base_url=settings.ollama_base_url)

    embedder = None
    if args.mode in {"vector", "hybrid"}:
        from jobrag.embed import get_embedder

        embedder = get_embedder(threads=4)
        embedder.dimension  # load before timing  # noqa: B018

    result = answer_question(
        question,
        llm=llm,
        embedder=embedder,
        k=args.k,
        mode=args.mode,
        use_filters=not args.no_filters,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("answer") else 1

    if result.get("error"):
        print(f"error: {result['error']}", file=sys.stderr)
        return 1
    print(f"\n{result['answer']}\n")
    print(f"filtres   {result['filters']}" + ("  (relâchés)" if result["filters_relaxed"] else ""))
    for note in result["notes"]:
        print(f"  note    {note}")
    print(f"sources   {result['sources_count']} offres, citées : {result['cited_indexes']}")
    for citation in result["citations"]:
        truncated = " [tronquée]" if citation["is_truncated"] else ""
        print(
            f"  [{citation['index']}] {citation['title'][:58]} · {citation['company'][:24]}"
            f" · {citation['city'] or '—'}{truncated}\n      {citation['url']}"
        )
    if result["invalid_citations"]:
        print(f"  CITATIONS INVALIDES : {result['invalid_citations']}")
    if result["uncited_answer"]:
        print("  ATTENTION : réponse sans aucune citation")
    timing = result["timing_ms"]
    usage = result["usage"]
    print(
        f"timing    {timing['total'] / 1000:.1f}s total "
        f"(retrieval {timing['retrieval']:.0f} ms, génération {timing['generation'] / 1000:.1f}s)"
        f" · {result['model']} · {usage['prompt_tokens']}+{usage['completion_tokens']} tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
