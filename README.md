# JobRAG

A question-answering assistant over real job offers, built to be **measured** rather than
demonstrated: every retrieval and answer-quality number in this README is produced by a script in
`scripts/`, and the command that produced it is printed next to it.

Each answer cites the offers it used (id, title, company, city, link), and answers
*"I don't know"* when the offers do not contain the answer.

**Status: phases 0–2a complete** — reproducible skeleton, a cleaned and deduplicated corpus of
32,013 offers exported read-only, and 51,321 chunks ready to embed. No retrieval results yet: the tables below stay empty until
phases 5 and 7 fill them from real runs.

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
make data                      # read-only export + corpus statistics + anonymised sample
make models                    # CPU-only torch + sentence-transformers (~2 GB, no GPU needed)
make schema                    # offers/chunks tables, HNSW + French full-text indexes
make index                     # embed the chunks (resumable)
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
  Asserted, not assumed: `tests/test_source_readonly_integration.py` checks that
  `transaction_read_only` is `on`, that `CREATE TABLE` raises `ReadOnlySqlTransaction`, and that
  `information_schema` exposes exactly one table to this role.
- Contact and CV-matching columns are never exported: the export uses an explicit allow-list of
  22 columns, so a new column upstream cannot leak in. `ApplyEmail` is set on 88 rows and
  `ContactName` on 1,294 rows of the source table; neither reaches `data/offers.jsonl`.
- E-mail addresses and phone numbers inside descriptions are masked before indexing: **186
  e-mails and 9 phone numbers** on the current snapshot, counted per offer in the export so the
  masking can be audited instead of trusted.
- `data/offers.jsonl` is git-ignored; only the small anonymised sample is committed. A
  `check-added-large-files` hook (512 KB) makes an accidental commit of the snapshot fail.
- Credentials live only in `.env` (git-ignored, mode 600). `config.mask_dsn()` is the single place
  that formats a DSN for display, and a test asserts no password reaches `/health`.

## The corpus

<!-- BEGIN: corpus-stats -->
<!-- generated by: `python scripts/corpus_stats.py` -->
**32013 rows** exported, **23715** canonical after deduplication
(8298 copies flagged with `duplicate_of`), **20186** truncated by the source
board (63%). Usable full-text corpus: **9089 offers**.
6228 companies, 2183 cities, posted between
2017-11-15 and 2026-11-15.
Masked before indexing: 186 e-mail addresses, 9 phone numbers.

| source | offers | canonical | truncated | truncated % | mean chars |
| --- | --- | --- | --- | --- | --- |
| adzuna | 19874 | 14361 | 19719 | 99% | 499 |
| france-travail | 11774 | 9007 | 465 | 4% | 2555 |
| greenhouse | 108 | 104 | 0 | 0% | 5456 |
| euraxess | 100 | 100 | 2 | 2% | 6785 |
| recruitee | 89 | 85 | 0 | 0% | 3417 |
| ashby | 55 | 46 | 0 | 0% | 8579 |
| welcome-to-the-jungle | 5 | 4 | 0 | 0% | 595 |
| abg | 4 | 4 | 0 | 0% | 5310 |
| unknown | 3 | 3 | 0 | 0% | 4797 |
| smartrecruiters | 1 | 1 | 0 | 0% | 3217 |

| description chars | min | p25 | p50 | p75 | p95 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full text | 18 | 1524 | 2379 | 3475 | 4782 | 17859 | 2554 |
| truncated at source | 393 | 500 | 500 | 500 | 500 | 9125 | 616 |

| contract | canonical offers |
| --- | --- |
| CDI | 11648 |
| non précisé | 7064 |
| CDD | 2539 |
| freelance | 1392 |
| intérim | 968 |
| doctorat | 104 |

Largest duplicate groups (same position, one row per city):

| copies | cities | title | company |
| --- | --- | --- | --- |
| 82 | 82 | Agent commercial / Agente commerciale en immobilier (H/F) | MEGAGENCE |
| 73 | 22 | Ingénieur Systèmes d'Informations Industriels H/F (CDI) | THALES SA |
| 71 | 71 | Développeur web fullstack (H/F) | LEIHIA |
| 66 | 66 | Développeur web fullstack H/F | Crédit Agricole |
| 57 | 7 | Ingénieur Développement Logiciels Embarqués F/H | Thales |
<!-- END: corpus-stats -->

## The phase-1 pipeline

`make export` streams the source table through a server-side cursor and writes one JSON object per
offer (32,013 rows, 80 MB, **14 s**). Each record is cleaned in a fixed order — mask contacts,
unwrap the rare HTML, normalise whitespace — then annotated:

- `is_truncated`: the source board cut the description (ellipsis, or exactly 500 characters).
- `duplicate_of` / `duplicate_group` / `duplicate_count`: near-duplicate grouping. Two rules,
  merged by union-find, so a chain of links forms one group: *same cleaned description + same
  company*, and *same normalised title + same company* (the same position posted once per city).
  Copies are kept rather than dropped — the canonical row carries `duplicate_cities`, because
  "where is this job open?" is a question the assistant has to answer.
- `title_norm` / `company_norm`: comparison keys (unaccented, lowercased, gender markers and legal
  forms removed). Never shown to the user.

## Chunking

<!-- BEGIN: chunking-stats -->
<!-- generated by: `python scripts/build_chunks.py --max-chars 1200 --overlap 150` -->
**23715 canonical offers → 51321 chunks** in 4.9s
(`max_chars=1200`, `overlap=150`, `min_chars=200`).
8298 flagged copies were not chunked. 16863 chunks
(33%) come from an offer truncated at the source and are flagged
as such. No chunk exceeds the budget: the longest is 1199 characters, header and overlap
included.

|  | min | p50 | p75 | p95 | max | mean |
| --- | --- | --- | --- | --- | --- | --- |
| chunks per offer | 1 | 1 | 3 | 6 | 23 | 2.2 |
| characters per chunk | 57 | 749 | 1023 | 1175 | 1199 | 788.6 |

| section | chunks | share |
| --- | --- | --- |
| (aucune) | 38813 | 76% |
| missions | 5651 | 11% |
| profil | 4268 | 8% |
| competences | 1586 | 3% |
| avantages | 492 | 1% |
| experience | 238 | 0% |
| entreprise | 151 | 0% |
| processus | 122 | 0% |
<!-- END: chunking-stats -->

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
| 1 | Read-only export, masking, deduplication, corpus statistics | **done** |
| 2a | Section-aware chunking with metadata | **done** |
| 2b | pgvector schema, embeddings, `/search` | next |
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
- **63% of the descriptions are truncated by the source board** (Adzuna cuts at 500 characters).
  After deduplication, the full-text corpus is **9,089 offers**, not 32,013. Truncated offers are
  exported and flagged: they can be cited for their metadata (city, contract, company, salary),
  but their text does not contain the requirements, so they cannot answer "what skills does this
  offer ask for?".
- Section headings (missions / profil / compétences) are recognisable in only **47%** of the
  full-text offers; the rest need size-based chunking.
- `PostedAt` has exactly one implausible value (2017); every other offer is dated 2026.
- A single language model and a single embedding model are evaluated end-to-end; the comparisons in
  phase 9 are partial by design.
- The corpus is a snapshot. No incremental re-indexing.

Author: Amine Sabri.
