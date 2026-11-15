"""The read-only guarantee on the source database, asserted rather than trusted.

Needs .env to point at the jobrag_ro role: `make test-all`.
"""

import psycopg
import pytest

from jobrag.config import get_settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def source_dsn() -> str:
    dsn = get_settings().source_db_dsn
    if not dsn:
        pytest.skip("SOURCE_DB_DSN not configured")
    return dsn


def test_the_session_is_read_only(source_dsn):
    with psycopg.connect(source_dsn, connect_timeout=10) as conn, conn.cursor() as cur:
        cur.execute("select current_user, current_setting('transaction_read_only')")
        user, read_only = cur.fetchone()
        assert read_only == "on", f"{user} can write to the source database"


def test_writing_is_refused(source_dsn):
    with (
        psycopg.connect(source_dsn, connect_timeout=10) as conn,
        conn.cursor() as cur,
        pytest.raises(psycopg.errors.ReadOnlySqlTransaction),
    ):
        cur.execute("create temporary table _probe (id int)")


def test_the_role_sees_only_the_offers_table(source_dsn):
    with psycopg.connect(source_dsn, connect_timeout=10) as conn, conn.cursor() as cur:
        cur.execute("""
            select distinct table_name from information_schema.columns
            where table_schema not in ('pg_catalog', 'information_schema')
        """)
        assert {row[0] for row in cur.fetchall()} == {"JobOffers"}
