"""Prompt construction, citation parsing, and the refusal paths."""

import pytest

from jobrag.filters import Filters
from jobrag.generate import (
    REFUSAL,
    SYSTEM_PROMPT,
    answer_question,
    build_prompt,
    format_sources,
    is_refusal,
    parse_citations,
    select_sources,
)
from jobrag.llm.base import Completion, LLMError


def chunk(offer_id, index=0, text="Missions : Kubernetes.", **extra):
    return {
        "chunk_id": f"{offer_id}:{index}",
        "offer_id": offer_id,
        "chunk_index": index,
        "text": f"Titre · ACME · Lille\n\n{text}",
        "header_chars": len("Titre · ACME · Lille") + 2,
        "title": "Ingénieur DevOps",
        "company": "ACME",
        "city": "Lille",
        "contract_label": "CDI",
        "salary": None,
        "url": f"https://example.org/offers/{offer_id}",
        "is_truncated": False,
        **extra,
    }


class FakeLLM:
    """Records the prompt it was given and returns a scripted answer."""

    name = "fake"

    def __init__(self, answer="Ingénieur DevOps chez ACME [1] demande Kubernetes.", error=None):
        self.answer = answer
        self.error = error
        self.calls: list[dict] = []

    def complete(self, prompt, *, system=None, temperature=0.0, max_tokens=250):
        self.calls.append({"prompt": prompt, "system": system})
        if self.error:
            raise self.error
        return Completion(
            text=self.answer, model="fake", latency_ms=1.0, prompt_tokens=10, completion_tokens=5
        )


def test_one_source_per_offer():
    # Retrieval returns three chunks of the same posting; the prompt must not
    # spend its budget on duplicates.
    hits = [chunk("a", 0), chunk("a", 1), chunk("a", 2), chunk("b", 0)]
    sources = select_sources(hits, max_sources=5)
    assert [source["offer_id"] for source in sources] == ["a", "b"]
    assert sources[0]["index"] == 1 and sources[1]["index"] == 2


def test_sources_respect_the_context_budget():
    hits = [chunk(str(i), 0, text="x" * 900) for i in range(10)]
    sources = select_sources(hits, max_sources=10, max_context_chars=1000, max_chars_per_source=500)
    assert len(sources) <= 3
    assert all(len(source["text"]) <= 500 for source in sources)


def test_max_sources_is_respected():
    hits = [chunk(str(i)) for i in range(10)]
    assert len(select_sources(hits, max_sources=3)) == 3


def test_a_truncated_source_is_flagged_in_the_prompt():
    sources = select_sources([chunk("a", is_truncated=True)])
    block = format_sources(sources)
    assert "tronquée" in block
    # And the rule about it is in the system prompt.
    assert "tronquée" in SYSTEM_PROMPT


def test_sources_are_numbered_from_one():
    block = format_sources(select_sources([chunk("a"), chunk("b")]))
    assert block.startswith("[1]")
    assert "\n[2] " in block or "[2]" in block


def test_the_prompt_puts_the_instruction_before_the_sources():
    # A user message ending in "Réponse :" made a 3B model answer "[1], [2]".
    prompt = build_prompt("Question ?", select_sources([chunk("a")]))
    assert prompt.index("réponds à la question") < prompt.index("Offres :")
    assert not prompt.rstrip().endswith(":")
    assert prompt.rstrip().endswith("Question ?")


def test_an_empty_source_list_asks_for_the_refusal():
    prompt = build_prompt("Question ?", [])
    assert REFUSAL in prompt


@pytest.mark.parametrize(
    ("answer", "valid", "invalid"),
    [
        ("A [1] et B [2].", [1, 2], []),
        ("A [1] et encore [1].", [1], []),
        ("A [9].", [], [9]),  # a fabricated reference
        ("A [1] et [9].", [1], [9]),
        ("Aucune citation.", [], []),
        ("Ordre inverse [2] puis [1].", [2, 1], []),
    ],
)
def test_citation_parsing(answer, valid, invalid):
    sources = select_sources([chunk("a"), chunk("b")])
    parsed = parse_citations(answer, sources)
    assert parsed["cited_indexes"] == valid
    assert parsed["invalid_citations"] == invalid
    assert [citation["index"] for citation in parsed["citations"]] == valid


@pytest.mark.parametrize(
    ("answer", "refused"),
    [
        ("Je ne sais pas.", True),
        ("je ne sais pas", True),
        ("I don't know.", True),
        ("Je ne peux pas répondre à partir des offres.", True),
        ("Les offres fournies ne contiennent pas cette information.", True),
        ("Ingénieur DevOps chez ACME [1].", False),
    ],
)
def test_refusal_detection(answer, refused):
    assert is_refusal(answer) is refused


def test_answer_question_returns_citations(monkeypatch):
    monkeypatch.setattr("jobrag.generate.retrieve", lambda *a, **k: [chunk("a"), chunk("b")])
    llm = FakeLLM()
    result = answer_question("Quelles offres Kubernetes ?", llm=llm, use_filters=False)
    assert result["refused"] is False
    assert result["cited_indexes"] == [1]
    assert result["citations"][0]["url"].startswith("https://")
    assert result["uncited_answer"] is False
    assert result["degenerate_answer"] is False
    assert result["sources_count"] == 2
    assert llm.calls[0]["system"] == SYSTEM_PROMPT


def test_an_answer_with_no_citation_is_flagged(monkeypatch):
    monkeypatch.setattr("jobrag.generate.retrieve", lambda *a, **k: [chunk("a")])
    result = answer_question(
        "Question ?", llm=FakeLLM(answer="Il y a trois offres à Lille."), use_filters=False
    )
    assert result["uncited_answer"] is True


def test_citations_without_prose_are_flagged_as_degenerate(monkeypatch):
    # What the first prompt produced: "[1], [2], [3]" and nothing else.
    monkeypatch.setattr("jobrag.generate.retrieve", lambda *a, **k: [chunk("a"), chunk("b")])
    result = answer_question("Question ?", llm=FakeLLM(answer="[1], [2]"), use_filters=False)
    assert result["degenerate_answer"] is True
    assert result["refused"] is False


def test_filters_that_match_nothing_refuse_without_calling_the_model(monkeypatch):
    """The measured hallucination: 'en alternance' matched zero offers, the
    filter was relaxed, and the model invented an alternance offer."""
    monkeypatch.setattr("jobrag.generate.retrieve", lambda *a, **k: [])
    monkeypatch.setattr(
        "jobrag.generate.extract_filters",
        lambda *a, **k: __import__("jobrag.filters", fromlist=["Extraction"]).Extraction(
            filters=Filters(contract_type=4), notes=[]
        ),
    )
    llm = FakeLLM()
    result = answer_question("offres en alternance ?", llm=llm, cities=set())
    assert result["refused"] is True
    assert result["citations"] == []
    assert llm.calls == [], "the model must not be called when the filter is the answer"
    assert result["timing_ms"]["generation"] == 0.0
    assert "contrat=4" in result["answer"]


def test_relaxing_filters_is_opt_in(monkeypatch):
    calls = {"n": 0}

    def fake_retrieve(*a, **k):
        calls["n"] += 1
        return [] if k.get("filters") and not k["filters"].is_empty() else [chunk("a")]

    monkeypatch.setattr("jobrag.generate.retrieve", fake_retrieve)
    monkeypatch.setattr(
        "jobrag.generate.extract_filters",
        lambda *a, **k: __import__("jobrag.filters", fromlist=["Extraction"]).Extraction(
            filters=Filters(contract_type=4), notes=[]
        ),
    )
    result = answer_question(
        "offres en alternance ?", llm=FakeLLM(), cities=set(), relax_filters=True
    )
    assert result["filters_relaxed"] is True
    assert any("without them" in note for note in result["notes"])


def test_a_backend_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr("jobrag.generate.retrieve", lambda *a, **k: [chunk("a")])
    result = answer_question(
        "Question ?", llm=FakeLLM(error=LLMError("ollama unreachable")), use_filters=False
    )
    assert result["answer"] is None
    assert "unreachable" in result["error"]
