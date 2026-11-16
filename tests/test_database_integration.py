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
        for field in ("chunk_id", "offer_id", "text", "title", "company", "url"):
            assert hit[field] is not None


def test_keyword_search_runs_against_the_real_database():
    from jobrag.retrieve.keyword import search_chunks

    out = search_chunks("kubernetes", k=3)
    assert out["results"], "no French full-text match for a term known to be in the corpus"
    assert all(isinstance(hit["score"], float) for hit in out["results"])


def test_a_natural_question_falls_back_to_a_loose_keyword_query():
    # Measured before the fix: websearch_to_tsquery ANDs every term, so this
    # question returned zero rows — no chunk contains "quelles" and "offres"
    # and "demandent" and "kubernetes" and "paris".
    from jobrag.retrieve.keyword import search_chunks

    question = "Quelles offres demandent Kubernetes à Paris ?"
    assert search_chunks(question, k=5, strict_only=True)["results"] == []

    out = search_chunks(question, k=5)
    assert out["query_mode"] == "strict+loose"
    assert len(out["results"]) == 5


def test_a_term_query_stays_strict():
    from jobrag.retrieve.keyword import search_chunks

    out = search_chunks("kubernetes", k=5)
    assert out["query_mode"] == "strict"
    assert len(out["results"]) == 5
    assert all("kubernetes" in hit["text"].lower() for hit in out["results"])
