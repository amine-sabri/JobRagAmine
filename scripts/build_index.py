#!/usr/bin/env python
"""Embed data/chunks.jsonl into the pgvector database.

Resumable: chunks that already carry an embedding are skipped, so an
interrupted run (or a 40-minute CPU run on a laptop) can be continued.

Usage:
    python scripts/build_index.py --limit 500      # measure throughput first
    python scripts/build_index.py                  # the full corpus
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrag.config import get_settings, mask_dsn  # noqa: E402
from jobrag.db import connection  # noqa: E402
from jobrag.embed import get_embedder  # noqa: E402
from jobrag.retrieve.vector import to_vector_literal  # noqa: E402

OFFER_COLUMNS = [
    "id",
    "source_name",
    "title",
    "company",
    "location",
    "city",
    "region",
    "country",
    "is_remote",
    "contract_type",
    "contract_label",
    "language_code",
    "status",
    "level",
    "salary",
    "url",
    "posted_at",
    "discovered_at",
    "expires_at",
    "is_truncated",
    "duplicate_cities",
    "duplicate_count",
    "description_chars",
]

UPSERT_OFFER = f"""
INSERT INTO offers ({", ".join(OFFER_COLUMNS)})
VALUES ({", ".join(f"%({c})s" for c in OFFER_COLUMNS)})
ON CONFLICT (id) DO UPDATE SET
    {", ".join(f"{c} = EXCLUDED.{c}" for c in OFFER_COLUMNS if c != "id")},
    indexed_at = now()
"""

INSERT_CHUNK = """
INSERT INTO chunks
    (chunk_id, offer_id, chunk_index, section, sections, text, n_chars, header_chars,
     is_truncated, embedding)
VALUES
    (%(chunk_id)s, %(offer_id)s, %(chunk_index)s, %(section)s, %(sections)s, %(text)s,
     %(n_chars)s, %(header_chars)s, %(is_truncated)s, %(embedding)s::vector)
ON CONFLICT (chunk_id) DO UPDATE SET
    text = EXCLUDED.text,
    n_chars = EXCLUDED.n_chars,
    header_chars = EXCLUDED.header_chars,
    section = EXCLUDED.section,
    sections = EXCLUDED.sections,
    is_truncated = EXCLUDED.is_truncated,
    embedding = EXCLUDED.embedding
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="data/chunks.jsonl", type=Path)
    parser.add_argument("--offers", default="data/offers.jsonl", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="only N chunks (for timing)")
    parser.add_argument("--batch-size", type=int, default=16, help="model forward batch")
    parser.add_argument(
        "--window",
        type=int,
        default=512,
        help="chunks handed to the model per call, and rows written per commit",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true", help="re-embed everything")
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="do not drop/rebuild the HNSW index around a large load",
    )
    args = parser.parse_args()

    for path in (args.chunks, args.offers):
        if not path.exists():
            print(f"{path} not found — run `make data` first", file=sys.stderr)
            return 2

    settings = get_settings()
    print(f"database   {mask_dsn(settings.jobrag_db_dsn)}")
    print(f"model      {settings.embedding_model} (cpu, {args.threads} threads)")

    with args.chunks.open(encoding="utf-8") as fh:
        chunks = [json.loads(line) for line in fh]
    if args.limit:
        chunks = chunks[: args.limit]
    wanted_offers = {chunk["offer_id"] for chunk in chunks}

    with args.offers.open(encoding="utf-8") as fh:
        offers = [json.loads(line) for line in fh if json.loads(line)["id"] in wanted_offers]

    started = time.perf_counter()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                UPSERT_OFFER,
                [{column: offer.get(column) for column in OFFER_COLUMNS} for offer in offers],
            )
            conn.commit()
            print(f"offers     {len(offers)} upserted in {time.perf_counter() - started:.1f}s")

            if not args.no_resume:
                cur.execute("SELECT chunk_id FROM chunks WHERE embedding IS NOT NULL")
                done = {row[0] for row in cur.fetchall()}
                before = len(chunks)
                chunks = [chunk for chunk in chunks if chunk["chunk_id"] not in done]
                if before != len(chunks):
                    print(f"resume     {before - len(chunks)} chunks already embedded, skipping")

            drop_index = not args.keep_index and len(chunks) > 5000
            if drop_index:
                cur.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
                conn.commit()
                print("hnsw       index dropped for the load (rebuilt at the end)")

        if not chunks:
            print("nothing to do")
            return 0

        # Sort by length before batching. A batch is padded to its longest
        # member, so mixing a 57-character chunk with a 1,199-character one
        # wastes most of the compute on padding tokens. Padding is masked, so
        # the embeddings are bit-for-bit the ones the unsorted order produces —
        # only the wasted work disappears.
        chunks.sort(key=lambda chunk: chunk["n_chars"])

        embedder = get_embedder(batch_size=args.batch_size, threads=args.threads)
        load_started = time.perf_counter()
        print(f"loading    {embedder.model_name} …")
        dimension = embedder.dimension
        print(
            f"            {dimension} dimensions, "
            f"loaded in {time.perf_counter() - load_started:.1f}s"
        )

        embed_seconds = 0.0
        write_seconds = 0.0
        processed = 0
        total = len(chunks)
        run_started = time.perf_counter()
        # One encode() call per window, not per batch. sentence-transformers
        # sorts the window by length internally and pads each forward batch to
        # its own longest member, which measured 1.8x faster than handing it
        # 16 chunks at a time (0.96 -> 1.75 chunks/s on a mid-corpus slice).
        with conn.cursor() as cur:
            for start in range(0, total, args.window):
                batch = chunks[start : start + args.window]
                t0 = time.perf_counter()
                vectors = embedder.encode_passages(
                    [chunk["text"] for chunk in batch], batch_size=args.batch_size
                )
                t1 = time.perf_counter()
                cur.executemany(
                    INSERT_CHUNK,
                    [
                        {
                            "chunk_id": chunk["chunk_id"],
                            "offer_id": chunk["offer_id"],
                            "chunk_index": chunk["chunk_index"],
                            "section": chunk["section"],
                            "sections": chunk["sections"],
                            "text": chunk["text"],
                            "n_chars": chunk["n_chars"],
                            "header_chars": chunk["header_chars"],
                            "is_truncated": chunk["is_truncated"],
                            "embedding": to_vector_literal(vector),
                        }
                        for chunk, vector in zip(batch, vectors, strict=True)
                    ],
                )
                conn.commit()
                t2 = time.perf_counter()
                embed_seconds += t1 - t0
                write_seconds += t2 - t1
                processed += len(batch)
                rate = processed / (time.perf_counter() - run_started)
                eta = (total - processed) / rate if rate else 0
                print(
                    f"  {processed:>6}/{total}  {rate:5.2f} chunks/s  eta {eta / 3600:4.1f} h",
                    flush=True,
                )

        with conn.cursor() as cur:
            if drop_index:
                index_started = time.perf_counter()
                print("hnsw       rebuilding index …", flush=True)
                cur.execute(
                    "CREATE INDEX chunks_embedding_idx ON chunks "
                    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
                )
                conn.commit()
                print(f"            built in {(time.perf_counter() - index_started) / 60:.1f} min")
            cur.execute("ANALYZE chunks")
            cur.execute("SELECT count(*), count(embedding) FROM chunks")
            rows, embedded = cur.fetchone()
            conn.commit()

    elapsed = time.perf_counter() - run_started
    print(f"chunks     {rows} rows, {embedded} embedded")
    print(
        f"timing     {elapsed / 60:.1f} min total — embedding {embed_seconds / 60:.1f} min, "
        f"writing {write_seconds / 60:.1f} min, {processed / elapsed:.1f} chunks/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
