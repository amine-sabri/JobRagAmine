import pytest

from jobrag.config import get_settings

TEST_DSN = "postgresql://jobrag:test-password@127.0.0.1:55433/jobrag"


@pytest.fixture
def settings_env(monkeypatch, tmp_path):
    """Isolate the tests from the developer's real .env.

    Settings read `.env` relative to the working directory, so the tests run from
    an empty directory: otherwise a value set locally (SOURCE_DB_DSN, a different
    model) decides whether the suite passes.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JOBRAG_DB_DSN", TEST_DSN)
    monkeypatch.delenv("SOURCE_DB_DSN", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
