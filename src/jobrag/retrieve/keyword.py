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

**The query is cleaned before Postgres sees it.** `websearch_to_tsquery` ANDs
every term, so *"Quelles offres demandent Kubernetes à Paris ?"* required one
chunk containing `quelles` AND `offres` AND `demandent` AND `kubernetes` AND
`paris` — measured: zero rows. Postgres's `french` dictionary does not treat
those as stopwords, so the question scaffolding is stripped here first
(`jobrag.text.content_tokens`), leaving `kubernetes paris`.

**Two stages.** Strict (AND over the content words) runs first; if it returns
fewer than k rows, the remaining slots come from a loose OR query. Without the
cleaning, the loose stage was actively harmful: "Quelles offres demandent du
Pulumi ?" let the frequency of "offres" outrank the four offers that actually
mention Pulumi, because `ts_rank_cd` has no IDF. Tokens still reach SQL as bound
parameters only — Postgres does the lexemising.
"""

import time
from typing import Any

from ..db import connection
from ..text import content_tokens

#: AND semantics: every term must be present. Precise, and empty on questions.
STRICT_QUERY = "websearch_to_tsquery('french', %(query)s)"

#: OR semantics over the lexemes Postgres extracts, stopwords already removed.
LOOSE_QUERY = """to_tsquery(
    'french',
    nullif(array_to_string(tsvector_to_array(to_tsvector('french', %(query)s)), ' | '), '')
)"""

SEARCH_SQL = """
WITH q AS (SELECT {tsquery} AS tsquery)
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


def _run(cur, tsquery_sql: str, query: str, limit: int) -> list[dict[str, Any]]:
    cur.execute(SEARCH_SQL.format(tsquery=tsquery_sql), {"query": query, "limit": limit})
    columns = [description.name for description in cur.description]
    return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def clean_query(query: str) -> str:
    """The searchable part of a question. Falls back to the raw text if empty."""
    return " ".join(content_tokens(query)) or query


def search_chunks(query: str, *, k: int = 5, strict_only: bool = False) -> dict[str, Any]:
    """Top-k chunks for `query` by French full-text ranking.

    Same shape as the vector search, so the two can be fused (phase 5) and
    compared without the caller caring which retriever produced a row.
    """
    started = time.perf_counter()
    searchable = clean_query(query)
    with connection() as conn, conn.cursor() as cur:
        rows = _run(cur, STRICT_QUERY, searchable, k)
        mode = "strict"
        if len(rows) < k and not strict_only:
            before = len(rows)
            seen = {row["chunk_id"] for row in rows}
            for row in _run(cur, LOOSE_QUERY, searchable, k * 3):
                if row["chunk_id"] not in seen:
                    rows.append(row)
                    seen.add(row["chunk_id"])
                if len(rows) == k:
                    break
            if len(rows) > before:
                mode = "strict+loose"
    elapsed = time.perf_counter() - started

    for row in rows:
        row["score"] = round(float(row["score"]), 6)
        # Postgres hands back a uuid.UUID, which is neither JSON-serialisable
        # nor comparable to the string ids in the evaluation set: the metrics
        # would have scored every question 0.0 without raising anything.
        row["offer_id"] = str(row["offer_id"])
        if row["posted_at"] is not None:
            row["posted_at"] = row["posted_at"].isoformat()
    return {
        "query": query,
        "searchable": searchable,
        "k": k,
        "query_mode": mode,
        "results": rows[:k],
        "timing_ms": {"database": round(elapsed * 1000, 1), "total": round(elapsed * 1000, 1)},
    }
