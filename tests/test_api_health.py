"""/health contract, with the database faked: these tests need no container."""

from fastapi.testclient import TestClient

from jobrag import api

UP = {
    "reachable": True,
    "server_version": "16.4",
    "extensions": {"vector": True, "unaccent": True, "pg_trgm": True},
    "error": None,
}
DOWN = {
    "reachable": False,
    "server_version": None,
    "extensions": {"vector": False, "unaccent": False, "pg_trgm": False},
    "error": "OperationalError: connection refused",
}
NO_EXTENSION = {**UP, "extensions": {"vector": False, "unaccent": True, "pg_trgm": True}}


def _client(monkeypatch, database):
    monkeypatch.setattr(api, "check_database", lambda *a, **k: database)
    return TestClient(api.app, raise_server_exceptions=False)


def test_health_ok(monkeypatch, settings_env):
    response = _client(monkeypatch, UP).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_is_503_when_the_database_is_down(monkeypatch, settings_env):
    response = _client(monkeypatch, DOWN).get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_health_is_503_when_pgvector_is_missing(monkeypatch, settings_env):
    # Reachable but unusable: an index build would fail later, so say so now.
    response = _client(monkeypatch, NO_EXTENSION).get("/health")
    assert response.status_code == 503


def test_health_never_returns_a_password(monkeypatch, settings_env):
    body = _client(monkeypatch, UP).get("/health").text
    assert "test-password" not in body
