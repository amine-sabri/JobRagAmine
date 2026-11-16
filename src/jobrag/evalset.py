"""The evaluation set: schema, local search, and coverage accounting.

Amine writes the 100 questions; this module exists to make that fast and to
keep the file machine-readable. The search here is deliberately keyword-based
and local (it reads data/offers.jsonl): labelling must not depend on the index
being built, and it must not be biased by the retriever under test — if the
candidates came only from the vector search, the labels would inherit its
blind spots and recall@5 would be flattered.
"""

import math
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .enums import CONTRACT_LABELS

#: Question families the evaluation set has to cover, with the target mix.
QUESTION_TYPES: dict[str, str] = {
    "fact": "an exact fact from one offer (salary, city, contract, a named technology)",
    "filter": "a filtered list (city, contract type, language, remote)",
    "aggregation": "a count or a ranking over many offers",
    "offer": "a question about one specific offer, by id",
    "none": "no answer in the corpus — the assistant must refuse",
}
TARGET_MIX: dict[str, int] = {
    "fact": 30,
    "filter": 25,
    "aggregation": 15,
    "offer": 15,
    "none": 15,
}

FILTER_RE = re.compile(r"\b(city|contract|lang|remote|source|truncated):(\S+)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9#+]+")

# A question is mostly grammar: "Quelles offres demandent Kubernetes ?" carries
# one content word. Dropping these keeps the candidate search on the signal.
STOPWORDS = frozenset(
    """
    a au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais me meme
    mes moi mon ne nos notre nous on ou par pas pour qu que quel quelle quelles quels qui sa se
    ses son sur ta te tes toi ton tu un une vos votre vous y est sont ont offre offres poste
    postes emploi emplois recherche recherches demandent demande combien quoi dont ainsi
    the a an and are as at be by for from has have in is it its of on or that this to was were
    what which who with job jobs offer offers position positions looking
    """.split()  # noqa: SIM905 - a readable block beats a 120-item list literal
)


def fold(text: str) -> str:
    """Lowercase, unaccented form used for every comparison."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(fold(text))


def parse_query(raw: str) -> tuple[str, dict[str, str]]:
    """Split `kubernetes city:Lille contract:CDI` into text and filters.

    >>> parse_query("kubernetes city:Lille")
    ('kubernetes', {'city': 'lille'})
    """
    filters = {key.lower(): value.lower() for key, value in FILTER_RE.findall(raw)}
    text = FILTER_RE.sub("", raw).strip()
    return re.sub(r"\s+", " ", text), filters


def _contract_matches(offer: dict[str, Any], wanted: str) -> bool:
    if wanted.isdigit():
        return offer.get("contract_type") == int(wanted)
    label = fold(offer.get("contract_label") or "")
    return label == fold(wanted) or fold(wanted) in label


TRUTHY = {"1", "true", "yes", "oui"}


def matches_filters(offer: dict[str, Any], filters: dict[str, str]) -> bool:
    """Does this offer satisfy every `key:value` filter? Unknown keys are ignored."""
    checks = {
        "city": lambda v: fold(offer.get("city") or "") == fold(v),
        "contract": lambda v: _contract_matches(offer, v),
        "lang": lambda v: (offer.get("language_code") or "") == v,
        "remote": lambda v: bool(offer.get("is_remote")) is (v in TRUTHY),
        "source": lambda v: v in fold(offer.get("source_name") or ""),
        "truncated": lambda v: bool(offer.get("is_truncated")) is (v in TRUTHY),
    }
    return all(check(filters[key]) for key, check in checks.items() if key in filters)


def content_tokens(text: str) -> list[str]:
    """Tokens worth searching on: no stopwords, no single characters."""
    return [t for t in tokenise(text) if t not in STOPWORDS and len(t) > 1]


def document_frequencies(folded_texts: list[str]) -> Counter:
    """How many offers contain each token. One pass, used for IDF weighting."""
    frequencies: Counter = Counter()
    for text in folded_texts:
        frequencies.update(set(_TOKEN_RE.findall(text)))
    return frequencies


def idf(frequencies: Counter, n_documents: int, token: str) -> float:
    """Inverse document frequency, smoothed.

    This is what makes "kubernetes" outweigh "paris": without it, two common
    words in a title beat the one rare word the question is actually about.
    """
    return math.log((n_documents + 1) / (frequencies.get(token, 0) + 1)) + 1.0


def score_offer(offer: dict[str, Any], tokens: list[str], weights: dict[str, float]) -> float:
    """Score an offer against query tokens, weighting rare tokens higher.

    A title hit counts triple and a company hit double, but the token's IDF
    dominates — matching the rare word matters more than where it matched.

    Not meant to be a good retriever: meant to be a *different* one, so the
    candidate list is not a copy of what the system under test would return.
    """
    if not tokens:
        return 1.0
    title = fold(offer.get("title") or "")
    body = offer.get("_folded") or ""
    company = fold(offer.get("company") or "")
    score = 0.0
    for token in tokens:
        in_title, in_body, in_company = token in title, token in body, token in company
        if in_title or in_body or in_company:
            field = 3.0 * in_title + 2.0 * in_company + 1.0 * in_body
            score += weights.get(token, 1.0) * field
    return score


class OfferSearch:
    """Keyword search over the exported offers, for labelling only.

    Folds the text and counts document frequencies once; a query then costs a
    single pass over the offers (~0.4 s for 23,715 offers), which is fast
    enough to type against and needs no index, no model and no database.
    """

    def __init__(self, offers: list[dict[str, Any]]) -> None:
        self.offers = offers
        for offer in self.offers:
            if "_folded" not in offer:
                offer["_folded"] = fold(
                    f"{offer.get('title') or ''} {offer.get('description') or ''}"
                )
        self.frequencies = document_frequencies([o["_folded"] for o in self.offers])
        self.n_documents = len(self.offers)

    def weights(self, tokens: list[str]) -> dict[str, float]:
        return {token: idf(self.frequencies, self.n_documents, token) for token in tokens}

    def search(self, raw_query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """Top `limit` offers for a raw query, filters (`city:Lille`) included."""
        text, filters = parse_query(raw_query)
        tokens = content_tokens(text)
        # A term already expressed as a filter must not also score as text:
        # "kubernetes à Paris en CDI" with city:Paris contract:CDI was ranking
        # any Paris CDI title above the one offer that mentions Kubernetes.
        redundant = {token for value in filters.values() for token in tokenise(value)}
        tokens = [token for token in tokens if token not in redundant]
        weights = self.weights(tokens)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for offer in self.offers:
            if not matches_filters(offer, filters):
                continue
            score = score_offer(offer, tokens, weights)
            if score:
                scored.append((score, offer.get("posted_at") or "", offer))
        # Two stable passes: newest first, then score first. Clearer than
        # packing a descending date into a single sort key.
        scored.sort(key=lambda row: row[1], reverse=True)
        scored.sort(key=lambda row: row[0], reverse=True)
        return [offer for _, _, offer in scored[:limit]]


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(loaded, list):
        raise ValueError(f"{path} should contain a list of questions")
    return loaded


def save_questions(path: Path, questions: list[dict[str, Any]]) -> None:
    """Write the whole file atomically: a Ctrl-C must not truncate the set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        questions,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def next_question_id(questions: list[dict[str, Any]]) -> str:
    """q001, q002, … continuing after the highest existing number."""
    highest = 0
    for question in questions:
        match = re.fullmatch(r"q(\d+)", str(question.get("id", "")))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"q{highest + 1:03d}"


def make_question(
    question_id: str,
    text: str,
    question_type: str,
    offer_ids: list[str],
    *,
    search_used: str = "",
    notes: str = "",
) -> dict[str, Any]:
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"unknown type {question_type!r}, expected one of {list(QUESTION_TYPES)}")
    if question_type == "none" and offer_ids:
        raise ValueError("a 'none' question must have no relevant offer")
    if question_type != "none" and not offer_ids:
        raise ValueError(f"a '{question_type}' question needs at least one relevant offer id")
    return {
        "id": question_id,
        "question": text.strip(),
        "type": question_type,
        "relevant_offer_ids": list(dict.fromkeys(offer_ids)),
        "search_used": search_used.strip(),
        "notes": notes.strip(),
        "created_at": date.today().isoformat(),
    }


def coverage(questions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per type: how many are written, and how many the target mix asks for."""
    written: dict[str, int] = dict.fromkeys(QUESTION_TYPES, 0)
    for question in questions:
        question_type = question.get("type")
        if question_type in written:
            written[question_type] += 1
    return {
        question_type: {
            "written": written[question_type],
            "target": TARGET_MIX[question_type],
            "missing": max(TARGET_MIX[question_type] - written[question_type], 0),
        }
        for question_type in QUESTION_TYPES
    }


AGGREGATION_MARKERS = (
    "combien",
    "nombre",
    "les plus",
    "le plus",
    "la plus",
    "top ",
    "classement",
    "repartition",
    "moyenne",
    "how many",
    "most common",
)
OFFER_MARKERS = ("cette offre", "cet offre", "offre [", "this offer", "pour cette")


def guess_type(question: str, filters: dict[str, str] | None = None, offer_ids: int = 0) -> str:
    """A default type for the labelling prompt, to be confirmed or overridden."""
    folded = fold(question)
    if any(marker in folded for marker in OFFER_MARKERS) or offer_ids == 1 and "[" in question:
        return "offer"
    if any(marker in folded for marker in AGGREGATION_MARKERS):
        return "aggregation"
    if filters or re.search(r"\b(a|en|dans|sur)\s+[A-Z]", question):
        return "filter"
    return "fact"


def describe_offer(offer: dict[str, Any], *, width: int = 70) -> str:
    """One compact line for the candidate list."""
    title = (offer.get("title") or "")[:width]
    bits = [offer.get("company") or "?", offer.get("city") or "—"]
    contract = offer.get("contract_label")
    if contract and contract != CONTRACT_LABELS[0]:
        bits.append(contract)
    if offer.get("is_remote"):
        bits.append("télétravail")
    if offer.get("is_truncated"):
        bits.append("TRONQUÉE")
    if offer.get("duplicate_count"):
        bits.append(f"+{offer['duplicate_count']} copies")
    return f"{title}  ·  {' · '.join(str(b) for b in bits)}"
