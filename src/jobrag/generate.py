"""Answering a question from retrieved offers, with citations or not at all.

The prompt carries four rules, and each exists because of a specific failure:

1. **Numbered sources, and a citation after every claim.** Without `[n]`
   markers an answer cannot be checked, and phase 7 has nothing to measure.
2. **"Je ne sais pas." when the sources do not contain the answer.** A model
   asked about something absent will otherwise produce a plausible offer that
   does not exist, which is the worst possible failure for a job search.
3. **Nothing may be inferred from a truncated description.** 63% of this corpus
   is cut at 500 characters by the source board, so "this offer does not require
   Kubernetes" is unknowable from a teaser. Truncated sources are marked as such
   in the prompt.
4. **One source per offer, and at most two chunks of it.** Retrieval happily
   returns three chunks of the same posting; spending the context window on
   duplicates crowds out the offers that would have answered.
"""

import re
import time
from typing import Any

from .filters import Extraction, Filters, extract_filters, known_cities
from .llm.base import LLM, LLMError
from .retrieve.fusion import fuse_hits
from .retrieve.keyword import search_chunks as keyword_search
from .retrieve.vector import search_chunks as vector_search

REFUSAL = "Je ne sais pas."

# Short on purpose. The first version listed six numbered rules (~350 tokens of
# the ~1,100-token prompt) and a 3B model answered "[1], [2], [3]" — citations
# with no sentence. On a CPU that prefills at a few tens of tokens per second,
# every rule also costs seconds per question. One example of a good answer
# teaches the format better than six rules describing it.
# Three iterations, each driven by what the model actually produced:
#   1. Six numbered rules (~350 of ~1,100 prompt tokens) -> "[1], [2], [3]":
#      citations with nothing asserted.
#   2. Rules trimmed, but the refusal clause sat next to the citation format and
#      the model templated it per source -> "[1] Je ne sais pas. [2] Je ne sais
#      pas."  So the refusal instruction now comes last, and says explicitly
#      that it replaces the whole answer.
#   3. A user message ending in "Réponse :" invited completion-style mimicry.
#      The instruction now comes first and the sources follow.
SYSTEM_PROMPT = """Tu réponds à des questions sur des offres d'emploi, uniquement à partir \
des offres numérotées qu'on te donne.

- Écris une à trois phrases complètes et factuelles. Nomme l'intitulé et l'entreprise, \
et place le numéro de l'offre juste après, comme ceci : « Ingénieur DevOps chez ACME [1] ».
- N'invente aucune offre et n'ajoute aucune connaissance extérieure.
- Une offre marquée « tronquée » ne dit rien des compétences exigées : cite-la seulement \
pour son intitulé, son entreprise, sa ville, son contrat ou son salaire.

Cas particulier : si aucune des offres fournies ne répond à la question, ta réponse \
entière doit être exactement « Je ne sais pas. », sans citation et sans rien ajouter."""

CITATION_RE = re.compile(r"\[(\d{1,2})\]")
REFUSAL_RE = re.compile(
    r"\bje ne sais pas\b|\bi (?:do not|don't) know\b|\bje ne peux pas r[eé]pondre\b"
    r"|\bles offres (?:fournies )?ne (?:contiennent|permettent)\b",
    re.IGNORECASE,
)

# Sized to the measured hardware, not to taste: on these 4 CPU cores
# qwen2.5:3b decodes at 4-6 tokens/s and prefills a few hundred tokens in
# seconds, so every 1,000 characters of context and every 50 tokens of answer
# cost real seconds. One chunk per offer, four offers, 700 characters each:
# ~800 prompt tokens, which keeps an answer near a minute instead of three.
MAX_CHUNKS_PER_OFFER = 1
MAX_CONTEXT_CHARS = 2000
MAX_CHARS_PER_SOURCE = 500
DEFAULT_MAX_TOKENS = 200


def select_sources(
    hits: list[dict[str, Any]],
    *,
    max_sources: int = 6,
    max_chunks_per_offer: int = MAX_CHUNKS_PER_OFFER,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    max_chars_per_source: int = MAX_CHARS_PER_SOURCE,
) -> list[dict[str, Any]]:
    """One numbered source per offer, best rank first, within a context budget."""
    per_offer: dict[str, list[dict[str, Any]]] = {}
    for hit in hits:
        chunks = per_offer.setdefault(hit["offer_id"], [])
        if len(chunks) < max_chunks_per_offer:
            chunks.append(hit)

    sources: list[dict[str, Any]] = []
    used_chars = 0
    for offer_id, chunks in per_offer.items():
        if len(sources) >= max_sources:
            break
        best = chunks[0]
        text = "\n\n".join(
            chunk["text"][chunk.get("header_chars", 0) :].strip() for chunk in chunks
        )
        text = (text.strip() or best["text"])[:max_chars_per_source]
        if used_chars + len(text) > max_context_chars and sources:
            continue
        used_chars += len(text)
        sources.append(
            {
                "index": len(sources) + 1,
                "offer_id": offer_id,
                "title": best.get("title"),
                "company": best.get("company"),
                "city": best.get("city"),
                "contract_label": best.get("contract_label"),
                "salary": best.get("salary"),
                "url": best.get("url"),
                "is_truncated": bool(best.get("is_truncated")),
                "chunks": [chunk["chunk_id"] for chunk in chunks],
                "text": text,
            }
        )
    return sources


def format_sources(sources: list[dict[str, Any]]) -> str:
    """The numbered block the model reads."""
    blocks = []
    for source in sources:
        header = " · ".join(
            str(part)
            for part in (
                source["title"],
                source["company"],
                source["city"] or "ville non précisée",
                source["contract_label"],
                (f"salaire : {source['salary']}" if source.get("salary") else None),
            )
            if part
        )
        note = " — description tronquée par la source" if source["is_truncated"] else ""
        blocks.append(f"[{source['index']}] {header}{note}\n{source['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, sources: list[dict[str, Any]]) -> str:
    """Instruction first, then the sources, then the question.

    A user message ending in "Réponse :" makes a small chat model imitate the
    pattern instead of answering — measured: it emitted "[1], [2], [3]".
    """
    if not sources:
        return (
            "Aucune offre n'a été trouvée pour cette question.\n\n"
            f"Question : {question}\n\n"
            f"Réponds exactement : « {REFUSAL} »"
        )
    return (
        "En t'appuyant uniquement sur les offres ci-dessous, réponds à la question en une à "
        "trois phrases, chaque offre mentionnée étant suivie de son numéro.\n\n"
        f"Offres :\n\n{format_sources(sources)}\n\n"
        f"Question : {question}"
    )


def parse_citations(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Which sources the answer cites, and which citations point nowhere.

    An out-of-range `[9]` matters: it is a fabricated reference, and the
    citation-accuracy metric of phase 7 counts it.
    """
    cited = [int(number) for number in CITATION_RE.findall(answer)]
    by_index = {source["index"]: source for source in sources}
    valid = [index for index in dict.fromkeys(cited) if index in by_index]
    invalid = [index for index in dict.fromkeys(cited) if index not in by_index]
    return {
        "cited_indexes": valid,
        "invalid_citations": invalid,
        "citations": [
            {
                key: by_index[index][key]
                for key in (
                    "index",
                    "offer_id",
                    "title",
                    "company",
                    "city",
                    "contract_label",
                    "url",
                    "is_truncated",
                )
            }
            for index in valid
        ],
    }


def is_refusal(answer: str) -> bool:
    """Did the model decline to answer?"""
    return bool(REFUSAL_RE.search(answer))


def retrieve(
    question: str,
    *,
    embedder: Any,
    k: int,
    mode: str,
    filters: Filters | None,
    ef_search: int = 80,
) -> list[dict[str, Any]]:
    depth = max(k * 4, 20)
    if mode == "keyword":
        return keyword_search(question, k=depth, filters=filters)["results"]
    if mode == "vector":
        return vector_search(question, embedder, k=depth, ef_search=ef_search, filters=filters)[
            "results"
        ]
    vector_hits = vector_search(question, embedder, k=depth, ef_search=ef_search, filters=filters)[
        "results"
    ]
    keyword_hits = keyword_search(question, k=depth, filters=filters)["results"]
    return fuse_hits([vector_hits, keyword_hits])


def answer_question(
    question: str,
    *,
    llm: LLM,
    embedder: Any = None,
    k: int = 3,
    mode: str = "hybrid",
    use_filters: bool = True,
    relax_filters: bool = False,
    cities: set[str] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Answer `question` from the index, with citations, or refuse.

    When extracted filters match nothing, the question is refused rather than
    answered from offers that ignore the criterion — see the comment below for
    the hallucination that forced this. `relax_filters=True` restores the old
    behaviour, so phase 7 can measure both.
    """
    started = time.perf_counter()
    extraction = Extraction(filters=Filters(), notes=[])
    if use_filters:
        extraction = extract_filters(
            question, cities=set(cities) if cities is not None else set(known_cities())
        )
    filters = extraction.filters
    notes = list(extraction.notes)

    hits = retrieve(question, embedder=embedder, k=k, mode=mode, filters=filters)
    relaxed = False
    if not hits and not filters.is_empty():
        if not relax_filters:
            # Measured hallucination: "en alternance" extracted contract=4,
            # which matches zero offers in this corpus. Relaxing the filter fed
            # the model four CDD/CDI offers and it answered "[2] Ingénieur
            # système Linux ... proposé une opportunité d'alternance" — an
            # invention, on precisely the question class where refusing matters.
            # When the filter IS the question, finding nothing is the answer, so
            # this returns deterministically without calling the model at all.
            return {
                "question": question,
                "answer": f"{REFUSAL} Aucune offre du corpus ne correspond à ces critères "
                f"({filters.describe()}).",
                "refused": True,
                "cited_indexes": [],
                "invalid_citations": [],
                "citations": [],
                "uncited_answer": False,
                "degenerate_answer": False,
                "sources": [],
                "sources_count": 0,
                "filters": filters.describe(),
                "filters_relaxed": False,
                "notes": [*notes, "no offer matches these filters: refused without generating"],
                "mode": mode,
                "k": k,
                "model": "none (deterministic refusal)",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "timing_ms": {
                    "retrieval": round((time.perf_counter() - started) * 1000, 1),
                    "generation": 0.0,
                    "total": round((time.perf_counter() - started) * 1000, 1),
                },
            }
        notes.append(f"no offer matched the filters ({filters.describe()}): searched without them")
        relaxed = True
        hits = retrieve(question, embedder=embedder, k=k, mode=mode, filters=None)
    retrieved_at = time.perf_counter()

    sources = select_sources(hits, max_sources=k)
    prompt = build_prompt(question, sources)
    try:
        completion = llm.complete(prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens)
    except LLMError as exc:
        return {
            "question": question,
            "error": str(exc),
            "answer": None,
            "sources_count": len(sources),
            "filters": filters.describe(),
            "notes": notes,
        }
    finished = time.perf_counter()

    parsed = parse_citations(completion.text, sources)
    refused = is_refusal(completion.text)
    # "[1], [2], [3]" is what a small model produces when the prompt is all
    # rules: citations with nothing asserted. It is not an answer, and phase 7
    # must not score it as one.
    prose = CITATION_RE.sub("", completion.text)
    degenerate = len(re.sub(r"[\s,.;:·\-]+", "", prose)) < 15
    return {
        "question": question,
        "answer": completion.text,
        "refused": refused,
        **parsed,
        # An answer that asserts something without citing anything is the
        # faithfulness risk phase 7 measures; surfaced here, not hidden.
        "uncited_answer": not refused and not parsed["cited_indexes"],
        "degenerate_answer": degenerate,
        "sources": [
            {
                key: source[key]
                for key in ("index", "offer_id", "title", "company", "city", "url", "is_truncated")
            }
            for source in sources
        ],
        "sources_count": len(sources),
        "filters": filters.describe(),
        "filters_relaxed": relaxed,
        "notes": notes,
        "mode": mode,
        "k": k,
        "model": completion.model,
        "usage": {
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
        },
        "timing_ms": {
            "retrieval": round((retrieved_at - started) * 1000, 1),
            "generation": round((finished - retrieved_at) * 1000, 1),
            "total": round((finished - started) * 1000, 1),
        },
    }
