"""/ask contract, with retrieval and generation faked."""

from fastapi.testclient import TestClient

from jobrag import api
from jobrag.llm import LLMError

ANSWER = {
    "question": "Quelles offres demandent Kubernetes à Lille ?",
    "answer": "Ingénieur DevOps chez ACME [1] demande Kubernetes.",
    "refused": False,
    "cited_indexes": [1],
    "invalid_citations": [],
    "citations": [
        {
            "index": 1,
            "offer_id": "a",
            "title": "Ingénieur DevOps",
            "company": "ACME",
            "city": "Lille",
            "contract_label": "CDI",
            "url": "https://example.org/offers/a",
            "is_truncated": False,
        }
    ],
    "uncited_answer": False,
    "degenerate_answer": False,
    "sources": [],
    "sources_count": 3,
    "filters": "ville=Lille",
    "filters_relaxed": False,
    "notes": [],
    "mode": "hybrid",
    "k": 3,
    "model": "qwen2.5:3b-instruct",
    "usage": {"prompt_tokens": 877, "completion_tokens": 74},
    "timing_ms": {"retrieval": 60.0, "generation": 113000.0, "total": 113060.0},
}


def _client(monkeypatch, payload=None, error=None):
    def fake_answer(question, **kwargs):
        if error is not None:
            raise error
        return {
            **(payload or ANSWER),
            "question": question,
            **{key: kwargs[key] for key in ("k", "mode") if key in kwargs},
        }

    monkeypatch.setattr(api, "answer_question", fake_answer)
    monkeypatch.setattr(api, "_embedder", lambda: object())
    monkeypatch.setattr(api, "_llm", lambda: object())
    return TestClient(api.app, raise_server_exceptions=False)


def test_ask_returns_an_answer_with_citations(monkeypatch, settings_env):
    response = _client(monkeypatch).get("/ask", params={"q": "offres kubernetes à Lille ?"})
    assert response.status_code == 200
    body = response.json()
    assert body["cited_indexes"] == [1]
    assert body["citations"][0]["url"].startswith("https://")
    assert body["refused"] is False
    # Every citation must carry what a reader needs to check it.
    for field in ("offer_id", "title", "company", "city", "url"):
        assert body["citations"][0][field] is not None


def test_ask_passes_k_and_mode_through(monkeypatch, settings_env):
    body = (
        _client(monkeypatch)
        .get("/ask", params={"q": "une question ?", "k": 5, "mode": "keyword"})
        .json()
    )
    assert (body["k"], body["mode"]) == (5, "keyword")


def test_keyword_mode_needs_no_embedding_model(monkeypatch, settings_env):
    def explode():
        raise AssertionError("the embedding model must not load for a keyword ask")

    monkeypatch.setattr(api, "answer_question", lambda question, **kwargs: {**ANSWER})
    monkeypatch.setattr(api, "_embedder", explode)
    monkeypatch.setattr(api, "_llm", lambda: object())
    response = TestClient(api.app).get("/ask", params={"q": "question ?", "mode": "keyword"})
    assert response.status_code == 200


def test_a_refusal_is_a_200_not_an_error(monkeypatch, settings_env):
    refusal = {
        **ANSWER,
        "answer": "Je ne sais pas.",
        "refused": True,
        "citations": [],
        "cited_indexes": [],
    }
    response = _client(monkeypatch, payload=refusal).get(
        "/ask", params={"q": "offres à Reykjavik ?"}
    )
    assert response.status_code == 200
    assert response.json()["refused"] is True


def test_an_unreachable_model_is_a_503(monkeypatch, settings_env):
    response = _client(monkeypatch, error=LLMError("ollama unreachable")).get(
        "/ask", params={"q": "une question ?"}
    )
    assert response.status_code == 503
    assert response.json()["error"] == "generation backend unavailable"


def test_a_backend_error_in_the_payload_is_also_a_503(monkeypatch, settings_env):
    broken = {"question": "q", "error": "ollama timed out", "answer": None}
    response = _client(monkeypatch, payload=broken).get("/ask", params={"q": "une question ?"})
    assert response.status_code == 503


def test_k_and_question_length_are_bounded(monkeypatch, settings_env):
    client = _client(monkeypatch)
    assert client.get("/ask", params={"q": "ab"}).status_code == 422
    assert client.get("/ask", params={"q": "question valable", "k": 0}).status_code == 422
    assert client.get("/ask", params={"q": "question valable", "k": 50}).status_code == 422
