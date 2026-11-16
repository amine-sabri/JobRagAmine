"""Needs the container: `make up`, then `make test-all`."""

import pytest

from jobrag.db import REQUIRED_EXTENSIONS, check_database

pytestmark = pytest.mark.integration


def test_database_is_reachable_and_has_pgvector():
    status = check_database()
    assert status["reachable"], status["error"]
    for extension in REQUIRED_EXTENSIONS:
        assert status["extensions"][extension], f"{extension} not installed"


def test_the_jobrag_role_can_write_its_own_database(tmp_path):
    # The mirror image of the read-only test on the source database: here we
    # *must* be able to write, since this is where the index lives.
    from jobrag.db import connection

    with connection() as conn, conn.cursor() as cur:
        cur.execute("CREATE TEMPORARY TABLE _smoke (id int)")
        cur.execute("INSERT INTO _smoke VALUES (1)")
        cur.execute("SELECT count(*) FROM _smoke")
        assert cur.fetchone()[0] == 1


class _StubEmbedder:
    """A query vector without a 1.1 GB model: this test is about the SQL."""

    dimension = 768

    def encode_queries(self, texts, **kwargs):
        return [[1.0] + [0.0] * 767 for _ in texts]

    def encode_passages(self, texts, **kwargs):
        return self.encode_queries(texts)


def test_vector_search_runs_against_the_real_database():
    # Regression: the mocked /search tests never executed this SQL, so
    # "SET LOCAL hnsw.ef_search = %s" (SET takes no bound parameters) only
    # failed the first time a real query ran.
    from jobrag.retrieve.vector import search_chunks

    out = search_chunks("développeur python", _StubEmbedder(), k=3, ef_search=40)
    assert out["k"] == 3
    assert out["ef_search"] == 40
    assert set(out["timing_ms"]) == {"embedding", "database", "total"}
    for hit in out["results"]:
        assert isinstance(hit["score"], float)
        # A uuid.UUID here is not JSON-serialisable and never equals the string
        # ids in eval/questions.yaml, so every metric would silently read 0.0.
        assert isinstance(hit["offer_id"], str)
        for field in ("chunk_id", "offer_id", "text", "title", "company", "url"):
            assert hit[field] is not None


def test_keyword_search_runs_against_the_real_database():
    from jobrag.retrieve.keyword import search_chunks

    out = search_chunks("kubernetes", k=3)
    assert out["results"], "no French full-text match for a term known to be in the corpus"
    assert all(isinstance(hit["score"], float) for hit in out["results"])


def test_a_natural_question_is_answered_without_the_loose_fallback():
    """The question scaffolding is stripped before Postgres sees the query.

    History: `websearch_to_tsquery` ANDs every term, so this question once
    required a chunk containing "quelles" AND "offres" AND "demandent" AND
    "kubernetes" AND "paris" and returned **zero** rows. The first fix added a
    loose OR fallback, which answered but ranked badly — `ts_rank_cd` has no
    IDF, so the frequency of "offres" buried the relevant offers. Cleaning the
    query to its content words makes the strict stage work, which is both
    precise and fast.
    """
    from jobrag.retrieve.keyword import clean_query, search_chunks

    question = "Quelles offres demandent Kubernetes à Paris ?"
    assert clean_query(question) == "kubernetes paris"

    out = search_chunks(question, k=5, strict_only=True)
    assert len(out["results"]) == 5, "the strict stage should answer on its own now"
    assert out["query_mode"] == "strict"
    assert out["searchable"] == "kubernetes paris"
    # AND semantics on the content words: every hit really has both.
    for hit in out["results"]:
        text = hit["text"].lower()
        assert "kubernetes" in text


def test_a_rare_technology_is_found_despite_the_question_wording():
    """The failure this fix was made for: Pulumi is in 4 of 23,715 offers."""
    from jobrag.retrieve.keyword import search_chunks

    out = search_chunks("Quelles offres demandent du Pulumi ?", k=5)
    assert out["results"], "no hit for a term known to be in the corpus"
    assert "pulumi" in out["results"][0]["text"].lower()


def test_a_term_query_stays_strict():
    from jobrag.retrieve.keyword import search_chunks

    out = search_chunks("kubernetes", k=5)
    assert out["query_mode"] == "strict"
    assert len(out["results"]) == 5
    assert all("kubernetes" in hit["text"].lower() for hit in out["results"])


def test_hybrid_search_runs_against_the_real_database():
    from jobrag.retrieve.fusion import fuse_hits
    from jobrag.retrieve.keyword import search_chunks as keyword_search
    from jobrag.retrieve.vector import search_chunks as vector_search

    query = "ingénieur devops kubernetes"
    vector_hits = vector_search(query, _StubEmbedder(), k=20)["results"]
    keyword_hits = keyword_search(query, k=20)["results"]
    fused = fuse_hits([vector_hits, keyword_hits], limit=5)

    assert len(fused) == 5
    for hit in fused:
        assert hit["rrf_score"] > 0
        assert len(hit["ranks"]) == 2
        assert any(rank is not None for rank in hit["ranks"])
    # Best first, and the payload survives the fusion.
    scores = [hit["rrf_score"] for hit in fused]
    assert scores == sorted(scores, reverse=True)
    assert all(hit["title"] and hit["url"] for hit in fused)


def test_the_search_endpoint_serves_all_three_modes_against_the_real_database(monkeypatch):
    """The whole path: HTTP -> retrieval -> SQL -> pgvector/tsvector -> JSON.

    Only the embedding model is stubbed. The mocked unit tests hid a crash in
    this exact SQL ("SET LOCAL hnsw.ef_search = %s"), so one test has to run it.
    """
    from fastapi.testclient import TestClient

    from jobrag import api

    monkeypatch.setattr(api, "_embedder", lambda: _StubEmbedder())
    client = TestClient(api.app)

    for mode in ("vector", "keyword", "hybrid"):
        response = client.get(
            "/search", params={"q": "ingénieur devops kubernetes", "k": 5, "mode": mode}
        )
        assert response.status_code == 200, (mode, response.text)
        body = response.json()
        assert body["results"], f"{mode} returned nothing"
        for hit in body["results"]:
            assert isinstance(hit["offer_id"], str), f"{mode}: offer_id is not a string"
            for field in ("chunk_id", "offer_id", "text", "title", "company", "url"):
                assert hit[field], f"{mode}: {field} missing from a citation"
        assert body["timing_ms"]["total"] > 0


def test_health_reports_a_fully_embedded_index():
    from jobrag.db import check_database

    status = check_database()
    assert status["index"]["chunks"] == status["index"]["embedded"], "some chunk has no vector"
    assert status["extensions"]["vector"] is True
