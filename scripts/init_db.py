#!/usr/bin/env python
"""Apply src/jobrag/schema.sql to the JobRAG database. Idempotent.

Usage:
    python scripts/init_db.py [--drop]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrag.config import get_settings, mask_dsn  # noqa: E402
from jobrag.db import connection  # noqa: E402

SCHEMA = Path(__file__).resolve().parents[1] / "src" / "jobrag" / "schema.sql"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drop",
        action="store_true",
        help="drop the chunks and offers tables first (asks for confirmation)",
    )
    args = parser.parse_args()

    settings = get_settings()
    print(f"target: {mask_dsn(settings.jobrag_db_dsn)}")

    if args.drop:
        answer = input("drop tables 'chunks' and 'offers'? type 'yes': ")
        if answer.strip() != "yes":
            print("aborted")
            return 1

    with connection() as conn, conn.cursor() as cur:
        if args.drop:
            cur.execute("DROP TABLE IF EXISTS chunks, offers CASCADE")
            print("dropped chunks, offers")
        cur.execute(SCHEMA.read_text(encoding="utf-8"))
        conn.commit()
        cur.execute("""
            select table_name, count(*)
            from information_schema.columns
            where table_schema = 'public' and table_name in ('offers', 'chunks')
            group by 1 order by 1
        """)
        for table, columns in cur.fetchall():
            cur.execute(f"select count(*) from {table}")  # noqa: S608 - fixed names
            print(f"  {table}: {columns} columns, {cur.fetchone()[0]} rows")
        cur.execute("""
            select indexname from pg_indexes
            where tablename in ('offers', 'chunks') order by indexname
        """)
        print("  indexes:", ", ".join(row[0] for row in cur.fetchall()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
