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


KEYWORD_HIT = {**HIT, "chunk_id": "b:0", "offer_id": "b", "score": 0.1608, "title": "Dev FastAPI"}


def _hybrid_client(monkeypatch):
    """Both retrievers faked: the endpoint's fusion wiring is what is tested."""
    monkeypatch.setattr(
        api,
        "search_chunks",
        lambda query, embedder, *, k=5, ef_search=80: {
            "query": query,
            "k": k,
            "results": [HIT, KEYWORD_HIT],
            "timing_ms": {"embedding": 40.0, "database": 5.0, "total": 45.0},
        },
    )
    monkeypatch.setattr(
        api,
        "keyword_search_chunks",
        lambda query, *, k=5: {
            "query": query,
            "k": k,
            "results": [KEYWORD_HIT],
            "timing_ms": {"database": 8.0, "total": 8.0},
        },
    )
    monkeypatch.setattr(api, "_embedder", lambda: object())
    return TestClient(api.app, raise_server_exceptions=False)


def test_keyword_mode_does_not_touch_the_embedding_model(monkeypatch, settings_env):
    def explode():
        raise AssertionError("the model must not be loaded for a keyword search")

    monkeypatch.setattr(api, "_embedder", explode)
    monkeypatch.setattr(
        api,
        "keyword_search_chunks",
        lambda query, *, k=5: {"query": query, "k": k, "results": [HIT], "timing_ms": {}},
    )
    response = TestClient(api.app).get("/search", params={"q": "kubernetes", "mode": "keyword"})
    assert response.status_code == 200
    assert response.json()["results"][0]["offer_id"] == HIT["offer_id"]


def test_hybrid_mode_fuses_both_runs(monkeypatch, settings_env):
    response = _hybrid_client(monkeypatch).get(
        "/search", params={"q": "kubernetes", "mode": "hybrid", "k": 2}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid"
    # The chunk both retrievers returned must come first: 2nd + 1st beats 1st alone.
    assert body["results"][0]["chunk_id"] == "b:0"
    assert body["results"][0]["ranks"] == [2, 1]
    assert body["results"][0]["rrf_score"] > body["results"][1]["rrf_score"]
    assert set(body["timing_ms"]) == {"vector", "keyword", "total"}


def test_hybrid_searches_deeper_than_k_before_fusing(monkeypatch, settings_env):
    # Fusion can only promote what the other list ranked low, so each retriever
    # must look past k.
    seen = {}

    monkeypatch.setattr(
        api,
        "search_chunks",
        lambda query, embedder, *, k=5, ef_search=80: (
            seen.update(vector_k=k)
            or {"query": query, "k": k, "results": [], "timing_ms": {"total": 1.0}}
        ),
    )
    monkeypatch.setattr(
        api,
        "keyword_search_chunks",
        lambda query, *, k=5: (
            seen.update(keyword_k=k)
            or {"query": query, "k": k, "results": [], "timing_ms": {"total": 1.0}}
        ),
    )
    monkeypatch.setattr(api, "_embedder", lambda: object())
    TestClient(api.app).get("/search", params={"q": "kubernetes", "mode": "hybrid", "k": 5})
    assert seen["vector_k"] == 20
    assert seen["keyword_k"] == 20


def test_an_unknown_mode_is_rejected(monkeypatch, settings_env):
    response = _client(monkeypatch).get("/search", params={"q": "kubernetes", "mode": "magic"})
    assert response.status_code == 422
