#!/usr/bin/env python
"""Build eval/probe_gold.yaml: a small gold set with *exhaustive* labels.

Why not label by hand from search results: the candidates would come from the
retrievers under test, so anything they all miss would never be labelled and
every score would be flattered. Here each question carries an objective
predicate over data/offers.jsonl — "description contains kubernetes and
terraform, city Paris, CDI" — and the relevant set is **every** canonical offer
satisfying it. Nothing is missed, and the labels are reproducible: this script
recomputes them.

Two deliberate choices keep it honest:

* **The question wording avoids the predicate's words** wherever a natural
  paraphrase exists ("orchestration de conteneurs" for kubernetes). A predicate
  is lexical, so a question repeating its terms would hand the keyword
  retriever the answer and understate the vector side.
* **A predicate matching more than 12 offers is rejected**, because a question
  with 600 right answers measures the labelling, not the retrieval.

This is a 15-question smoke set, not the evaluation set: with n=15 the
confidence interval on recall@5 is ±13 points, so it is for catching breakage
and comparing configurations roughly — not for a headline number.

Usage:
    python scripts/build_probe_gold.py
"""

import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrag.evalset import make_question, save_questions  # noqa: E402

MAX_RELEVANT = 12


def fold(text: str | None) -> str:
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@lru_cache(maxsize=256)
def _word(term: str) -> re.Pattern[str]:
    """`term` as a whole word.

    Substring matching produced false labels and cost the retrievers recall for
    being right: "erlang" matched **Erlangen** (a German city), "flink" matched
    "STAFFLINK RH" and the Dutch "een flinke dosis". Lookarounds rather than
    \b, so terms like `as/400` and `c++` still work.
    """
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")


@lru_cache(maxsize=256)
def _prefix(stem: str) -> re.Pattern[str]:
    """`stem` starting a word: `plomb` matches `plombier`, not `aplomb`."""
    return re.compile(rf"(?<!\w){re.escape(stem)}\w*")


def has_all(offer: dict, *terms: str) -> bool:
    return all(_word(term).search(offer["_text"]) for term in terms)


def has_any(offer: dict, *terms: str) -> bool:
    return any(_word(term).search(offer["_text"]) for term in terms)


def has_prefix(offer: dict, *stems: str) -> bool:
    return any(_prefix(stem).search(offer["_text"]) for stem in stems)


def city_is(offer: dict, name: str) -> bool:
    return fold(offer.get("city")) == fold(name)


# (id, question asked, type, criterion in words, predicate)
#
# Every criterion below was checked against the corpus first: the twelve
# conjunctions I tried by intuition ("kubernetes AND terraform in Paris",
# "fastapi AND postgres", "kafka AND spark") each matched 20 to 169 offers and
# were rejected by the guard below. Rare technologies are what produce an
# enumerable relevant set in a corpus this dense.
PROBES: list[tuple[str, str, str, str, object]] = [
    (
        "p001",
        "Quelle offre demande du développement en Elixir ?",
        "fact",
        "elixir in text (1 offer)",
        lambda o: has_all(o, "elixir"),
    ),
    (
        "p002",
        "Quelles offres demandent de l'Erlang ?",
        "none",
        "erlang as a whole word — zero offers. The two substring matches were "
        "'Erlangen', a German city, which is why this is a refusal question",
        lambda o: has_all(o, "erlang"),
    ),
    (
        "p003",
        "Quelles offres de robotique utilisent le simulateur Gazebo ?",
        "filter",
        "gazebo in text (2)",
        lambda o: has_all(o, "gazebo"),
    ),
    (
        "p004",
        "Quelles offres demandent du Pulumi ?",
        "filter",
        "pulumi in text (4)",
        lambda o: has_all(o, "pulumi"),
    ),
    (
        "p005",
        "Quelles offres demandent un service mesh pour Kubernetes ?",
        "filter",
        "PARAPHRASED — istio OR 'service mesh' in text (7); the question says "
        "neither 'istio' alone nor the product name only",
        lambda o: has_any(o, "istio", "service mesh"),
    ),
    (
        "p006",
        "Quelles offres font du traitement de flux avec Apache Flink ?",
        "filter",
        "flink as a whole word (3; substring matching had added STAFFLINK and a Dutch 'flinke')",
        lambda o: has_all(o, "flink"),
    ),
    (
        "p007",
        "Quelles offres embarquées travaillent sur le bus CAN d'un véhicule ?",
        "filter",
        "PARAPHRASED — 'can bus' OR 'bus can' in text (9)",
        lambda o: has_any(o, "can bus", "bus can"),
    ),
    (
        "p008",
        "Quelles offres utilisent Kubeflow pour industrialiser des modèles ?",
        "filter",
        "kubeflow in text (9)",
        lambda o: has_all(o, "kubeflow"),
    ),
    (
        "p009",
        "Quelles offres concernent un système AS/400 ?",
        "filter",
        "as/400 in text (10)",
        lambda o: has_all(o, "as/400"),
    ),
    (
        "p010",
        "Quelles offres de calcul scientifique demandent du Fortran ?",
        "filter",
        "fortran in text (11)",
        lambda o: has_all(o, "fortran"),
    ),
    (
        "p011",
        "Quelles offres utilisent la base de données Neo4j ?",
        "filter",
        "neo4j in text (12)",
        lambda o: has_all(o, "neo4j"),
    ),
    (
        "p012",
        "Quelles offres parlent d'exporter des modèles au format ONNX ?",
        "filter",
        "onnx in text (12)",
        lambda o: has_all(o, "onnx"),
    ),
    # Questions the corpus cannot answer: the refusal cases.
    (
        "p013",
        "Quelles offres proposent un contrat en alternance ?",
        "none",
        "contract=alternance(4) — measured: zero in the whole corpus",
        lambda o: o["contract_type"] == 4,
    ),
    (
        "p014",
        "Quelles offres sont basées à Reykjavik ?",
        "none",
        "city=Reykjavik — not a city in this corpus",
        lambda o: city_is(o, "Reykjavik"),
    ),
    (
        "p015",
        "Quelles offres recherchent un plombier spécialisé en informatique quantique ?",
        "none",
        "plomberie AND quantique — a deliberately absent combination",
        lambda o: has_prefix(o, "plomb") and has_any(o, "quantique", "quantum"),
    ),
]


def main() -> int:
    path = Path("data/offers.jsonl")
    if not path.exists():
        print(f"{path} not found — run `make export` first", file=sys.stderr)
        return 2

    offers = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            offer = json.loads(line)
            if offer["duplicate_of"] is not None:
                continue
            offer["_text"] = fold(f"{offer['title']} {offer['description']}")
            offers.append(offer)
    print(f"{len(offers)} canonical offers\n")

    questions = []
    problems = []
    for question_id, text, question_type, criterion, predicate in PROBES:
        matching = [offer for offer in offers if predicate(offer)]
        ids = sorted(offer["id"] for offer in matching)
        status = "ok"
        if question_type == "none" and ids:
            status = f"REJECTED: expected no answer, found {len(ids)}"
        elif question_type != "none" and not ids:
            status = "REJECTED: no offer matches the criterion"
        elif len(ids) > MAX_RELEVANT:
            status = f"REJECTED: {len(ids)} relevant offers (> {MAX_RELEVANT}), not measurable"
        print(f"{question_id}  {len(ids):>3} relevant  {status:<52} {criterion}")
        if status != "ok":
            problems.append((question_id, status))
            continue
        questions.append(
            make_question(
                question_id,
                text,
                question_type,
                ids,
                notes=f"exhaustive labels from: {criterion}",
            )
        )

    out = Path("eval/probe_gold.yaml")
    save_questions(out, questions)
    answerable = [q for q in questions if q["type"] != "none"]
    sizes = sorted(len(q["relevant_offer_ids"]) for q in answerable)
    print(f"\n{len(questions)} questions written to {out}")
    print(f"  answerable {len(answerable)}, refusals {len(questions) - len(answerable)}")
    if sizes:
        print(
            f"  relevant offers per answerable question: min {sizes[0]}, median "
            f"{sizes[len(sizes) // 2]}, max {sizes[-1]}"
        )
    if problems:
        names = ", ".join(question_id for question_id, _ in problems)
        print(f"  {len(problems)} rejected: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
