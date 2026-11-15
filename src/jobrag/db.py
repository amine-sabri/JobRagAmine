"""Access to JobRAG's own Postgres (pgvector).

Phase 0 opens one short-lived connection per call: simple, and fast enough while
there is no query path yet. A psycopg connection pool arrives in phase 3, when
/search makes per-request latency something worth measuring.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg

from .config import get_settings

#: Extensions the index relies on (installed by docker/init/01_extension.sql).
REQUIRED_EXTENSIONS = ("vector",)
OPTIONAL_EXTENSIONS = ("unaccent", "pg_trgm")


@contextmanager
def connection(dsn: str | None = None, *, read_only: bool = False) -> Iterator[psycopg.Connection]:
    """Connect to the JobRAG database (or to `dsn` if given)."""
    target = dsn or get_settings().jobrag_db_dsn
    with psycopg.connect(target, connect_timeout=5) as conn:
        if read_only:
            conn.read_only = True
        yield conn


def check_database(dsn: str | None = None) -> dict[str, Any]:
    """Report reachability, server version and installed extensions.

    Never raises, and never includes a password in its output: it feeds /health.
    """
    wanted = REQUIRED_EXTENSIONS + OPTIONAL_EXTENSIONS
    try:
        with connection(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT current_setting('server_version')")
            row = cur.fetchone()
            version = row[0] if row else None
            cur.execute("SELECT extname FROM pg_extension WHERE extname = ANY(%s)", (list(wanted),))
            installed = {name for (name,) in cur.fetchall()}
    except Exception as exc:  # noqa: BLE001 - /health must answer, whatever happens
        message = str(exc).strip()
        first_line = message.splitlines()[0] if message else exc.__class__.__name__
        return {
            "reachable": False,
            "server_version": None,
            "extensions": dict.fromkeys(wanted, False),
            "error": f"{exc.__class__.__name__}: {first_line}"[:300],
        }
    return {
        "reachable": True,
        "server_version": version,
        "extensions": {name: name in installed for name in wanted},
        "error": None,
    }
