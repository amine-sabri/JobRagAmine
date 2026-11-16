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
WHERE c.embedding IS NOT NULL
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
) -> dict[str, Any]:
    """Top-k chunks for `query`, with the latency split that explains them.

    `ef_search` is HNSW's accuracy/speed knob: higher explores more of the
    graph. It is reported with the results, because a recall number without it
    is not reproducible.
    """
    started = time.perf_counter()
    vector = embedder.encode_queries([query])[0]
    embedded = time.perf_counter()

    with connection() as conn, conn.cursor() as cur:
        # SET does not accept bound parameters, so the knob goes through
        # set_config(..., is_local => true), which does. Interpolating the
        # value into a SET statement would work too, and would be the kind of
        # string-built SQL this project should not contain.
        cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(int(ef_search)),))
        cur.execute(
            SEARCH_SQL,
            {"vector": to_vector_literal(vector), "limit": k},
        )
        columns = [description.name for description in cur.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
    finished = time.perf_counter()

    for row in rows:
        row["score"] = round(float(row["score"]), 4)
        if row["posted_at"] is not None:
            row["posted_at"] = row["posted_at"].isoformat()
    return {
        "query": query,
        "k": k,
        "ef_search": ef_search,
        "results": rows,
        "timing_ms": {
            "embedding": round((embedded - started) * 1000, 1),
            "database": round((finished - embedded) * 1000, 1),
            "total": round((finished - started) * 1000, 1),
        },
    }
