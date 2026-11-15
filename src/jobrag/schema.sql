-- JobRAG index schema. Idempotent: `python scripts/init_db.py` can be re-run.
--
-- Two tables on purpose. Offer metadata is written once and joined at query
-- time, so a chunk row carries no duplicated city or contract label: a
-- re-chunking with a different size rewrites `chunks` and leaves `offers`
-- untouched, which is what phase 5 needs to compare chunk sizes cheaply.

CREATE TABLE IF NOT EXISTS offers (
    id                  uuid PRIMARY KEY,
    source_name         text        NOT NULL,
    title               text        NOT NULL,
    company             text        NOT NULL,
    location            text,
    city                text,
    region              text,
    country             text,
    is_remote           boolean     NOT NULL DEFAULT false,
    contract_type       smallint    NOT NULL,
    contract_label      text,
    language_code       text        NOT NULL,
    status              smallint,
    level               smallint,
    salary              text,
    url                 text        NOT NULL,
    posted_at           timestamptz,
    discovered_at       timestamptz,
    expires_at          timestamptz,
    is_truncated        boolean     NOT NULL DEFAULT false,
    duplicate_cities    text[],
    duplicate_count     integer     NOT NULL DEFAULT 0,
    description_chars   integer     NOT NULL DEFAULT 0,
    indexed_at          timestamptz NOT NULL DEFAULT now()
);

-- Filters the questions actually use (phase 6): city, contract, date, language.
CREATE INDEX IF NOT EXISTS offers_city_idx ON offers (lower(city));
CREATE INDEX IF NOT EXISTS offers_contract_idx ON offers (contract_type);
CREATE INDEX IF NOT EXISTS offers_posted_idx ON offers (posted_at DESC);
CREATE INDEX IF NOT EXISTS offers_language_idx ON offers (language_code);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        text        PRIMARY KEY,
    offer_id        uuid        NOT NULL REFERENCES offers (id) ON DELETE CASCADE,
    chunk_index     integer     NOT NULL,
    section         text,
    sections        text[],
    text            text        NOT NULL,
    n_chars         integer     NOT NULL,
    header_chars    integer     NOT NULL,
    is_truncated    boolean     NOT NULL DEFAULT false,
    embedding       vector(768),
    -- French full-text vector, maintained by Postgres itself. Added now rather
    -- than in phase 5 so enabling hybrid search costs no re-indexing.
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('french', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_offer_idx ON chunks (offer_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

-- HNSW with cosine distance: e5 embeddings are L2-normalised, so cosine and
-- inner product rank identically, and HNSW needs no training step unlike
-- IVFFlat. Built after the rows are loaded — building it first would slow every
-- insert down for nothing.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
