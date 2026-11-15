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
