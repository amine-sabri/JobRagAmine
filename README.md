# JobRAG

A question-answering assistant over real job offers, built to be **measured** rather than
demonstrated: every retrieval and answer-quality number in this README is produced by a script in
`scripts/`, and the command that produced it is printed next to it.

Each answer cites the offers it used (id, title, company, city, link), and answers
*"I don't know"* when the offers do not contain the answer.

**Status: phase 0 complete** (reproducible skeleton). No retrieval results yet — the tables below
stay empty until phases 5 and 7 fill them from real runs.

## Architecture

```
                                 ┌──────────────────────────────┐
  JobCrusader Postgres           │  JobRAG Postgres 16 +        │
  (read-only role, never         │  pgvector 0.8.7  (Docker)    │
   written to)                   │                              │
        │                        │  offers    chunks            │
        │  export (phase 1)      │  embeddings (vector)         │
        ▼                        │  tsvector (French FTS)       │
  data/offers.jsonl  ──────────► │                              │
  mask e-mails/phones            └──────────────┬───────────────┘
  dedupe, normalise titles                      │
                                                │ retrieve
  chunk (phase 2) ──────────────────────────────┤  vector → +BM25/RRF → +reranker
                                                │
                                                ▼
                                    FastAPI  /search  /ask  /health
                                                │
                                                ▼
                                    Ollama (qwen2.5:3b-instruct)
                                    behind a swappable LLM interface
```

Dashed parts of the pipeline arrive phase by phase; see the table at the bottom.

## Quickstart

Requires Python 3.11+, Docker with Compose v2, and (for generation, from phase 6) Ollama.

```bash
cp .env.example .env          # then set JOBRAG_DB_PASSWORD and JOBRAG_DB_DSN
make install                  # venv + pinned dependencies + pre-commit hook
make up                       # pgvector container, waits until healthy
make test                     # unit tests, no database needed
make test-all                 # + the tests that need the container
make api                      # http://127.0.0.1:8010
make health                   # pretty-printed /health
```

`make` with no target lists every target.

### Verified on this machine

```console
$ make up
 Container jobrag-db Healthy

$ curl -s http://127.0.0.1:8010/health
{
  "status": "ok",
  "version": "0.1.0",
  "database": {
    "reachable": true,
    "server_version": "16.16 (Debian 16.16-1.pgdg12+1)",
    "extensions": {"vector": true, "unaccent": true, "pg_trgm": true},
    "error": null
  },
  "config": {
    "jobrag_db_dsn": "postgresql://jobrag:***@127.0.0.1:55433/jobrag",
    "source_db_dsn": "(not configured)",
    "embedding_model": "intfloat/multilingual-e5-base",
    "reranker_model": "BAAI/bge-reranker-base",
    "ollama_base_url": "http://127.0.0.1:11434",
    "ollama_model": "qwen2.5:3b-instruct"
  }
}

$ docker stop jobrag-db && curl -s -o /dev/null -w '%{http_code}\n' .../health
503            # status "degraded", error "OperationalError: connection refused"

$ make test-all
15 passed in 0.58s
```

`/health` returns **503** when the database is unreachable or `pgvector` is missing, so Compose
healthchecks and CI can rely on the status code instead of parsing the body.

## Layout

| Path | Role |
| --- | --- |
| `docker/` | Compose file + `init/01_extension.sql` (`vector`, `unaccent`, `pg_trgm`) |
| `src/jobrag/` | library code: `config.py`, `db.py`, `api.py`, then chunking / retrieval / generation |
| `scripts/` | every number in this README comes from one of these |
| `eval/questions.yaml` | the hand-written evaluation set (phase 4) |
| `eval/results/` | committed run outputs, one file per run |
| `data/` | `offers.jsonl` (git-ignored snapshot) + `sample_offers.jsonl` (~50 anonymised, committed) |
| `tests/` | one test file per non-trivial module; integration tests marked and excluded by default |

## Data, privacy and safety

- The source database is **read-only**: the export uses a dedicated `jobrag_ro` role whose only
  privilege is `SELECT` on `public."JobOffers"`, with `default_transaction_read_only = on`. The
  guarantee is a property of the role, not of the code being correct.
  ```sql
  CREATE ROLE jobrag_ro LOGIN PASSWORD '<secret>' NOSUPERUSER NOCREATEDB NOCREATEROLE;
  GRANT CONNECT ON DATABASE "<jobcrusader_db>" TO jobrag_ro;
  GRANT USAGE  ON SCHEMA public TO jobrag_ro;
  GRANT SELECT ON TABLE public."JobOffers" TO jobrag_ro;
  ALTER ROLE jobrag_ro SET default_transaction_read_only = on;
  ALTER ROLE jobrag_ro SET statement_timeout = '60s';
  ```
  A test asserting that an `INSERT` with this role fails will be added with phase 1.
- Applicant and contact fields are never exported. E-mail addresses and phone numbers inside
  descriptions are masked by regex before indexing.
- `data/offers.jsonl` is git-ignored; only the small anonymised sample is committed. A
  `check-added-large-files` hook (512 KB) makes an accidental commit of the snapshot fail.
- Credentials live only in `.env` (git-ignored, mode 600). `config.mask_dsn()` is the single place
  that formats a DSN for display, and a test asserts no password reaches `/health`.

## Results

<!-- BEGIN: retrieval-results (regenerated by `make report`) -->
_Empty until phase 5: recall@5, MRR and latency for vector-only, hybrid+RRF and hybrid+reranker._
<!-- END: retrieval-results -->

<!-- BEGIN: answer-results (regenerated by `make report`) -->
_Empty until phase 7: faithfulness (LLM judge + 30 hand-checked answers), correct-refusal rate,
citation accuracy._
<!-- END: answer-results -->

## Phases

| # | Phase | Status |
| --- | --- | --- |
| 0 | Reproducible skeleton: Compose, env, Makefile, ruff + pre-commit, pytest, `/health` | **done** |
| 1 | Read-only export, masking, deduplication, corpus statistics | next |
| 2 | Section-aware chunking with metadata | |
| 3 | Vector-only baseline, `/search` | |
| 4 | 100-question evaluation set, hand-labelled | |
| 5 | Retrieval metrics: baseline → hybrid + RRF → + reranker, with latency | |
| 6 | Generation: numbered sources, mandatory citations, metadata filters | |
| 7 | Answer evaluation: faithfulness, refusals, citation accuracy | |
| 8 | Packaging: diagram, results, failure cases, CI | |
| 9 | Optional: embedding-model comparison, caching, web page | |

## Known limits

- **No GPU on the development machine** (checked with `nvidia-smi`), 4 CPU cores, 7.6 GB RAM. This
  drives the model choices: `multilingual-e5-base` (278M) rather than `bge-m3` (568M), and
  `qwen2.5:3b-instruct` rather than the 7B variant. Indexing and query latency are reported as
  measured on this hardware, not on a GPU.
- A single language model and a single embedding model are evaluated end-to-end; the comparisons in
  phase 9 are partial by design.
- The corpus is a snapshot. No incremental re-indexing.

Author: Amine Sabri.
