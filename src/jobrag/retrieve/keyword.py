"""Keyword retrieval with Postgres full-text search (French configuration).

Why Postgres FTS rather than `rank_bm25`:

* the filters, the vectors and the keywords live in one engine, so a hybrid
  query with `city = 'Lille'` is one SQL statement and not three systems to
  keep in sync;
* `to_tsvector('french', …)` brings French stemming and stopwords, so
  "développeurs" matches "développeur", which a naive BM25 over raw tokens
  would miss;
* the `tsvector` is a generated column — Postgres maintains it, there is no
  second index to rebuild when a chunk changes.

The cost is honest: `ts_rank_cd` is **not** BM25. It weights term frequency and
proximity but has no document-length saturation, so very long chunks can score
higher than they deserve. If the phase-5 numbers show that, the comparison
against `rank_bm25` is the experiment to run.
"""

import time
from typing import Any

from ..db import connection

SEARCH_SQL = """
WITH q AS (SELECT websearch_to_tsquery('french', %(query)s) AS tsquery)
SELECT
    c.chunk_id,
    c.offer_id,
    c.chunk_index,
    c.section,
    c.text,
    c.is_truncated,
    ts_rank_cd(c.tsv, q.tsquery) AS score,
    o.title,
    o.company,
    o.city,
    o.contract_label,
    o.language_code,
    o.salary,
    o.url,
    o.posted_at,
    o.duplicate_count
FROM chunks c
JOIN offers o ON o.id = c.offer_id,
     q
WHERE c.tsv @@ q.tsquery
ORDER BY score DESC, c.chunk_id
LIMIT %(limit)s
"""


def search_chunks(query: str, *, k: int = 5) -> dict[str, Any]:
    """Top-k chunks for `query` by French full-text ranking.

    Same shape as the vector search, so the two can be fused (phase 5) and
    compared without the caller caring which retriever produced a row.
    """
    started = time.perf_counter()
    with connection() as conn, conn.cursor() as cur:
        cur.execute(SEARCH_SQL, {"query": query, "limit": k})
        columns = [description.name for description in cur.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
    elapsed = time.perf_counter() - started

    for row in rows:
        row["score"] = round(float(row["score"]), 6)
        if row["posted_at"] is not None:
            row["posted_at"] = row["posted_at"].isoformat()
    return {
        "query": query,
        "k": k,
        "results": rows,
        "timing_ms": {"database": round(elapsed * 1000, 1), "total": round(elapsed * 1000, 1)},
    }
