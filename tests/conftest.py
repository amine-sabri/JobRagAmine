import pytest

from jobrag.config import get_settings

TEST_DSN = "postgresql://jobrag:test-password@127.0.0.1:55433/jobrag"


@pytest.fixture
def settings_env(monkeypatch):
    """Isolate the tests from the developer's real .env."""
    monkeypatch.setenv("JOBRAG_DB_DSN", TEST_DSN)
    monkeypatch.delenv("SOURCE_DB_DSN", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
