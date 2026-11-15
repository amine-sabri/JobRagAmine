#!/usr/bin/env python
"""Export job offers from the source platform to data/offers.jsonl.

Read-only by construction: the connection refuses to start unless the session
is in a read-only transaction, and the columns are an explicit allow-list — no
`SELECT *`, so a new contact column added upstream can never leak into the
export.

Usage:
    python scripts/export_offers.py [--limit N] [--out data/offers.jsonl]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrag.config import get_settings  # noqa: E402
from jobrag.dedup import mark_duplicates  # noqa: E402
from jobrag.export import COLUMNS, EXCLUDED, build_offer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="export only N offers (debugging)")
    parser.add_argument("--out", default="data/offers.jsonl", type=Path)
    parser.add_argument("--itersize", type=int, default=2000)
    args = parser.parse_args()

    dsn = get_settings().source_db_dsn
    if not dsn:
        print("SOURCE_DB_DSN is not set in .env", file=sys.stderr)
        return 2

    started = time.perf_counter()
    offers: list[dict] = []
    with psycopg.connect(dsn, connect_timeout=10) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("select current_user, current_setting('transaction_read_only')")
            user, read_only = cur.fetchone()
            if read_only != "on":
                print(f"refusing to run: {user} is not in a read-only session", file=sys.stderr)
                return 3
            cur.execute(
                """select count("ApplyEmail") as apply_email, count("ContactName") as contact_name
                   from public."JobOffers" """
            )
            contact_counts = dict(zip(["ApplyEmail", "ContactName"], cur.fetchone(), strict=True))

        quoted = ", ".join(f'"{c}"' for c in COLUMNS)
        sql = f'select {quoted} from public."JobOffers" order by "DiscoveredAt", "Id"'
        if args.limit:
            sql += f" limit {int(args.limit)}"
        # Server-side cursor: 32k rows stream instead of landing in memory at once.
        with conn.cursor(name="offers_export", row_factory=psycopg.rows.dict_row) as cur:
            cur.itersize = args.itersize
            cur.execute(sql)
            for row in cur:
                offers.append(build_offer(row))

    fetched = time.perf_counter()
    stats = mark_duplicates(offers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for offer in offers:
            fh.write(json.dumps(offer, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(args.out)

    elapsed = time.perf_counter() - started
    emails = sum(o["masked_emails"] for o in offers)
    phones = sum(o["masked_phones"] for o in offers)
    truncated = sum(1 for o in offers if o["is_truncated"])
    print(f"user                 {user} (read-only session)")
    print(f"offers exported      {len(offers)}")
    print(f"columns kept         {len(COLUMNS)}  excluded {len(EXCLUDED)}: {', '.join(EXCLUDED)}")
    print(
        f"contact fields left  ApplyEmail set in {contact_counts['ApplyEmail']} rows, "
        f"ContactName in {contact_counts['ContactName']} rows — never exported"
    )
    print(f"masked in text       {emails} e-mails, {phones} phone numbers")
    print(f"truncated at source  {truncated} ({truncated / max(len(offers), 1):.1%})")
    print(
        f"duplicate groups     {stats['multi_member_groups']} groups, "
        f"{stats['duplicates']} copies flagged (rules: {stats['rule_hits']})"
    )
    print(f"largest group        {stats['largest_groups'][0] if stats['largest_groups'] else '-'}")
    print(f"output               {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"timing               {fetched - started:.1f}s fetch+clean, {elapsed:.1f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
