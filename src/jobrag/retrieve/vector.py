"""Vectors-only retrieval: the baseline every later improvement is measured against.

Deliberately the simplest thing that works — one HNSW nearest-neighbour query,
no keyword matching, no reranking. Its weaknesses are the point: exact tokens
("Kubernetes", "FastAPI") are where cosine similarity is weakest, and phase 5
has to prove it fixes them.
"""

import time
from typing import Any

from ..db import connection
from ..embed import Embedder
from ..filters import Filters

SEARCH_SQL = """
SELECT
    c.chunk_id,
    c.offer_id,
    c.chunk_index,
    c.section,
    c.text,
    c.is_truncated,
    1 - (c.embedding <=> %(vector)s::vector) AS score,
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
JOIN offers o ON o.id = c.offer_id
WHERE c.embedding IS NOT NULL{extra}
ORDER BY c.embedding <=> %(vector)s::vector
LIMIT %(limit)s
"""


def to_vector_literal(vector: Any) -> str:
    """pgvector's text input format, e.g. '[0.12,-0.03,...]'."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vector) + "]"


def search_chunks(
    query: str,
    embedder: Embedder,
    *,
    k: int = 5,
    ef_search: int = 80,
    filters: Filters | None = None,
) -> dict[str, Any]:
    """Top-k chunks for `query`, with the latency split that explains them.

    `ef_search` is HNSW's accuracy/speed knob: higher explores more of the
    graph. It is reported with the results, because a recall number without it
    is not reproducible.

    With metadata filters the plain index scan is not enough: HNSW returns its
    `ef_search` nearest neighbours and *then* the filter removes some, so a
    narrow filter can yield almost nothing. pgvector 0.8 solves this with
    iterative scans — the index keeps walking until enough rows survive the
    filter — so `hnsw.iterative_scan` is switched on only when a filter is
    present, and left off otherwise to keep the unfiltered path fast.
    """
    started = time.perf_counter()
    vector = embedder.encode_queries([query])[0]
    embedded = time.perf_counter()

    conditions, filter_params = (filters or Filters()).as_conditions()
    extra = ("\n  AND " + "\n  AND ".join(conditions)) if conditions else ""

    with connection() as conn, conn.cursor() as cur:
        # SET does not accept bound parameters, so the knob goes through
        # set_config(..., is_local => true), which does. Interpolating the
        # value into a SET statement would work too, and would be the kind of
        # string-built SQL this project should not contain.
        cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(int(ef_search)),))
        if conditions:
            cur.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
        cur.execute(
            SEARCH_SQL.format(extra=extra),
            {"vector": to_vector_literal(vector), "limit": k, **filter_params},
        )
        columns = [description.name for description in cur.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
    finished = time.perf_counter()

    for row in rows:
        row["score"] = round(float(row["score"]), 4)
        # Postgres hands back a uuid.UUID, which is neither JSON-serialisable
        # nor comparable to the string ids in the evaluation set: the metrics
        # would have scored every question 0.0 without raising anything.
        row["offer_id"] = str(row["offer_id"])
        if row["posted_at"] is not None:
            row["posted_at"] = row["posted_at"].isoformat()
    return {
        "query": query,
        "k": k,
        "ef_search": ef_search,
        "filters": (filters or Filters()).describe(),
        "results": rows,
        "timing_ms": {
            "embedding": round((embedded - started) * 1000, 1),
            "database": round((finished - embedded) * 1000, 1),
            "total": round((finished - started) * 1000, 1),
        },
    }
