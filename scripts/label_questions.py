#!/usr/bin/env python
"""Write and label the evaluation questions (phase 4), in the terminal.

Type a question, pick the offers that answer it from a numbered list, confirm
the type. The file is rewritten after every question, so Ctrl-C costs nothing.

The candidate search is keyword-based and local: it needs neither the index nor
the model, and — more importantly — it is *not* the retriever being evaluated.
Labelling from the system under test would hide exactly the offers that system
fails to find, and recall@5 would be flattered.

Usage:
    python scripts/label_questions.py            # continue the set
    python scripts/label_questions.py --stats    # coverage only, no prompt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrag.evalset import (  # noqa: E402
    QUESTION_TYPES,
    OfferSearch,
    coverage,
    describe_offer,
    guess_type,
    load_questions,
    make_question,
    next_question_id,
    parse_query,
    save_questions,
)

HELP = """
At the question prompt
  <your question>      start a new question
  /list [n]            the last n questions (default 10)
  /del q007            delete a question
  /stats               coverage against the target mix
  /help                this text
  /quit                save and leave

While choosing the offers
  1 3 5                these candidates answer the question
  /s <query>           search again — filters: city:Lille contract:CDI lang:en
                       remote:true truncated:false  (contract takes 1-7 or CDI/CDD/…)
  /more                ten more candidates
  /show 2              the full text of candidate 2
  /id 2                print candidate 2's offer id (and fill [id] in your question)
  /none                no offer answers this — a refusal question
  /skip                drop this question

Question types
""" + "\n".join(f"  {name:<12} {description}" for name, description in QUESTION_TYPES.items())


def load_offers(path: Path) -> list[dict]:
    """Canonical offers only: a copy is never the answer, its canonical row is."""
    with path.open(encoding="utf-8") as fh:
        return [
            offer
            for offer in (json.loads(line) for line in fh)
            if offer.get("duplicate_of") is None
        ]


def print_coverage(questions: list[dict], target: int) -> None:
    rows = coverage(questions)
    parts = [f"{name} {counts['written']}/{counts['target']}" for name, counts in rows.items()]
    missing = sum(counts["missing"] for counts in rows.values())
    print(f"\n{len(questions)}/{target} questions · " + " · ".join(parts))
    if missing:
        short = [name for name, counts in rows.items() if counts["missing"]]
        print(f"still short on: {', '.join(short)}")


def print_candidates(hits: list[dict], offset: int = 0) -> None:
    if not hits:
        print("  no candidate — try /s with other words, or /none")
        return
    for number, offer in enumerate(hits, start=offset + 1):
        print(f"  {number:>2}. {describe_offer(offer)}")


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "/quit"


def label_one(
    searcher: OfferSearch, question_text: str, question_id: str, limit: int
) -> dict | None:
    """Run the candidate loop for one question. None means skip."""
    query = question_text
    hits = searcher.search(query, limit=limit)
    print_candidates(hits)
    selected: list[dict] = []

    while True:
        answer = ask("offers> ")
        if answer in {"/skip", "/quit"}:
            return None
        if answer == "/none":
            return make_question(
                question_id, parse_query(question_text)[0], "none", [], search_used=query
            )
        if answer.startswith("/s "):
            query = answer[3:].strip()
            hits = searcher.search(query, limit=limit)
            print_candidates(hits)
            continue
        if answer == "/more":
            more = searcher.search(query, limit=len(hits) + limit)[len(hits) :]
            print_candidates(more, offset=len(hits))
            hits = hits + more
            continue
        if answer.startswith("/show "):
            index = _as_index(answer, hits)
            if index is not None:
                offer = hits[index]
                print(f"\n--- {offer['title']} · {offer['company']} · {offer.get('city')} ---")
                print(f"{offer['url']}\n")
                print(offer["description"][:2000])
                print("---\n")
            continue
        if answer.startswith("/id "):
            index = _as_index(answer, hits)
            if index is not None:
                offer_id = hits[index]["id"]
                print(f"  {offer_id}")
                if "[id]" in question_text:
                    question_text = question_text.replace("[id]", offer_id)
                    print(f"  question is now: {question_text}")
            continue
        if answer in {"/help", "?"}:
            print(HELP)
            continue

        numbers = [part for part in answer.replace(",", " ").split() if part.isdigit()]
        if not numbers:
            print("  expected numbers (e.g. '1 3'), /s, /more, /show, /id, /none or /skip")
            continue
        chosen = [hits[int(n) - 1] for n in numbers if 1 <= int(n) <= len(hits)]
        if not chosen:
            print(f"  numbers must be between 1 and {len(hits)}")
            continue
        selected = chosen
        for offer in selected:
            print(f"  kept: {describe_offer(offer)[:90]}")
        break

    default_type = guess_type(question_text, parse_query(query)[1], len(selected))
    while True:
        raw = ask(f"type [{'/'.join(QUESTION_TYPES)}] (enter = {default_type})> ")
        question_type = raw or default_type
        notes = ask("note (optional)> ")
        try:
            return make_question(
                question_id,
                parse_query(question_text)[0],
                question_type,
                [offer["id"] for offer in selected],
                search_used=query,
                notes=notes,
            )
        except ValueError as error:
            print(f"  {error}")


def _as_index(answer: str, hits: list[dict]) -> int | None:
    parts = answer.split()
    if len(parts) != 2 or not parts[1].isdigit():
        print("  usage: /show 2")
        return None
    index = int(parts[1]) - 1
    if not 0 <= index < len(hits):
        print(f"  numbers must be between 1 and {len(hits)}")
        return None
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offers", default="data/offers.jsonl", type=Path)
    parser.add_argument("--questions", default="eval/questions.yaml", type=Path)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--stats", action="store_true", help="print coverage and exit")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.stats:
        print_coverage(questions, args.target)
        return 0

    offers_path = args.offers
    if not offers_path.exists():
        fallback = Path("data/sample_offers.jsonl")
        if not fallback.exists():
            print(f"{offers_path} not found — run `make export` first", file=sys.stderr)
            return 2
        print(f"{offers_path} not found, using {fallback} (50 offers)")
        offers_path = fallback

    offers = load_offers(offers_path)
    print(f"indexing {len(offers)} canonical offers for the candidate search …", flush=True)
    searcher = OfferSearch(offers)
    print(f"ready. {args.questions} · /help for the commands")

    while True:
        print_coverage(questions, args.target)
        question_id = next_question_id(questions)
        text = ask(f"\n[{question_id}] question> ")
        if text in {"/quit", "/q", "quit", "exit"}:
            break
        if not text:
            continue
        if text in {"/help", "?"}:
            print(HELP)
            continue
        if text == "/stats":
            continue
        if text.startswith("/list"):
            parts = text.split()
            count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            for question in questions[-count:]:
                ids = len(question["relevant_offer_ids"])
                print(
                    f"  {question['id']} [{question['type']:<11}] "
                    f"{ids} offers · {question['question']}"
                )
            continue
        if text.startswith("/del "):
            wanted = text.split()[1]
            before = len(questions)
            questions = [q for q in questions if q["id"] != wanted]
            if len(questions) == before:
                print(f"  no question {wanted}")
            else:
                save_questions(args.questions, questions)
                print(f"  deleted {wanted}")
            continue
        if text.startswith("/"):
            print("  unknown command — /help")
            continue

        question = label_one(searcher, text, question_id, args.candidates)
        if question is None:
            print("  skipped")
            continue
        questions.append(question)
        save_questions(args.questions, questions)
        print(
            f"  saved {question['id']} [{question['type']}] "
            f"with {len(question['relevant_offer_ids'])} offer(s)"
        )

    save_questions(args.questions, questions)
    print_coverage(questions, args.target)
    print(f"\nwritten to {args.questions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
