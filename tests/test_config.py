import pytest
from pydantic import ValidationError

from jobrag.config import Settings, get_settings, mask_dsn

from .conftest import TEST_DSN


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "postgresql://jobrag:s3cret@127.0.0.1:55433/jobrag",
            "postgresql://jobrag:***@127.0.0.1:55433/jobrag",
        ),
        # No port.
        (
            "postgresql://ro:pw@db.internal/jobcrusader",
            "postgresql://ro:***@db.internal/jobcrusader",
        ),
        # Nothing to hide: returned untouched.
        (
            "postgresql://jobrag@127.0.0.1:55433/jobrag",
            "postgresql://jobrag@127.0.0.1:55433/jobrag",
        ),
        ("", ""),
        # A password containing an @ still gets masked (the last @ separates the host).
        ("postgresql://u:p@ss@127.0.0.1/db", "postgresql://u:***@127.0.0.1/db"),
    ],
)
def test_mask_dsn_hides_the_password(raw, expected):
    assert mask_dsn(raw) == expected


def test_mask_dsn_never_echoes_an_unparsable_dsn():
    # An invalid port makes urlsplit raise on .port; we must not fall through
    # and print the raw string, which could contain a password.
    assert mask_dsn("postgresql://u:p@host:not-a-port/db") == "<unparsable dsn>"


def test_settings_require_the_database_dsn():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_defaults_are_the_cpu_choices(settings_env):
    settings = get_settings()
    assert settings.jobrag_db_dsn == TEST_DSN
    assert settings.embedding_model == "intfloat/multilingual-e5-base"
    assert settings.ollama_model == "qwen2.5:3b-instruct"


def test_public_dict_leaks_no_password(settings_env):
    public = get_settings().public_dict()
    assert "test-password" not in str(public)
    assert public["source_db_dsn"] == "(not configured)"
