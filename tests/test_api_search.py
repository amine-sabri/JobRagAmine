"""/search contract, with retrieval faked: no model and no database needed."""

from fastapi.testclient import TestClient

from jobrag import api

HIT = {
    "chunk_id": "3f1b2c4d-0000-4000-8000-000000000001:0",
    "offer_id": "3f1b2c4d-0000-4000-8000-000000000001",
    "chunk_index": 0,
    "section": "competences",
    "text": "Dev Python · ACME · Lille · CDI\n\nKubernetes, FastAPI.",
    "is_truncated": False,
    "score": 0.8123,
    "title": "Développeur Python (H/F)",
    "company": "ACME",
    "city": "Lille",
    "contract_label": "CDI",
    "language_code": "fr",
    "salary": None,
    "url": "https://example.org/offers/1",
    "posted_at": "2026-10-01T08:30:00+00:00",
    "duplicate_count": 0,
}
PAYLOAD = {
    "query": "offres kubernetes",
    "k": 5,
    "ef_search": 80,
    "results": [HIT],
    "timing_ms": {"embedding": 41.2, "database": 7.8, "total": 49.0},
}


def _client(monkeypatch, result=None, error=None):
    def fake_search(query, embedder, *, k=5, ef_search=80):
        if error is not None:
            raise error
        return {**(result or PAYLOAD), "query": query, "k": k, "ef_search": ef_search}

    monkeypatch.setattr(api, "search_chunks", fake_search)
    monkeypatch.setattr(api, "_embedder", lambda: object())
    return TestClient(api.app, raise_server_exceptions=False)


def test_search_returns_chunks_with_scores_and_timings(monkeypatch, settings_env):
    response = _client(monkeypatch).get("/search", params={"q": "offres kubernetes"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "offres kubernetes"
    assert body["results"][0]["score"] == 0.8123
    # Every hit must carry what a citation needs.
    for field in ("offer_id", "title", "company", "city", "url"):
        assert body["results"][0][field] is not None
    assert set(body["timing_ms"]) == {"embedding", "database", "total"}


def test_k_and_ef_search_are_passed_through(monkeypatch, settings_env):
    response = _client(monkeypatch).get(
        "/search", params={"q": "data engineer", "k": 20, "ef_search": 200}
    )
    body = response.json()
    assert (body["k"], body["ef_search"]) == (20, 200)


def test_k_is_bounded(monkeypatch, settings_env):
    client = _client(monkeypatch)
    assert client.get("/search", params={"q": "x" * 5, "k": 0}).status_code == 422
    assert client.get("/search", params={"q": "x" * 5, "k": 500}).status_code == 422


def test_a_too_short_query_is_rejected(monkeypatch, settings_env):
    assert _client(monkeypatch).get("/search", params={"q": "a"}).status_code == 422


def test_a_missing_embedding_backend_is_a_503_not_a_500(monkeypatch, settings_env):
    # torch/sentence-transformers absent: the API must say so, not crash.
    client = _client(monkeypatch, error=ImportError("No module named 'sentence_transformers'"))
    response = client.get("/search", params={"q": "kubernetes"})
    assert response.status_code == 503
    assert response.json()["error"] == "embedding backend unavailable"
