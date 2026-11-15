"""Typed settings, loaded from the environment / .env.

Two rules this module exists to enforce:
  * no credential is ever hard-coded (they come from .env, which is git-ignored);
  * no credential is ever printed — anything user-facing goes through mask_dsn().
"""

from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def mask_dsn(dsn: str) -> str:
    """Return `dsn` with its password replaced by ***, safe to log or display.

    >>> mask_dsn("postgresql://jobrag:s3cret@127.0.0.1:55433/jobrag")
    'postgresql://jobrag:***@127.0.0.1:55433/jobrag'
    """
    if not dsn:
        return ""
    try:
        parts = urlsplit(dsn)
        password = parts.password
        port = parts.port
    except ValueError:
        # Unparsable: better to show nothing than to risk leaking a password.
        return "<unparsable dsn>"
    if not password:
        return dsn
    netloc = f"{parts.username or ''}:***@{parts.hostname or ''}"
    if port:
        netloc += f":{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class Settings(BaseSettings):
    """Everything the application needs to know, validated at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # JobRAG's own database (pgvector). Required: failing loudly beats failing late.
    jobrag_db_dsn: str = Field(min_length=1)

    # JobCrusader source database. Read-only role only; empty until phase 1.
    source_db_dsn: str = ""

    embedding_model: str = "intfloat/multilingual-e5-base"
    reranker_model: str = "BAAI/bge-reranker-base"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b-instruct"

    def public_dict(self) -> dict[str, str]:
        """Settings as they may be shown to a user: DSNs masked."""
        return {
            "jobrag_db_dsn": mask_dsn(self.jobrag_db_dsn),
            "source_db_dsn": mask_dsn(self.source_db_dsn) or "(not configured)",
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings. Call get_settings.cache_clear() in tests."""
    return Settings()
