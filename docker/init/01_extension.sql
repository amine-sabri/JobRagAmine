-- Runs once, on first start of an empty data volume.
CREATE EXTENSION IF NOT EXISTS vector;
-- unaccent + pg_trgm are used later for title normalisation / near-duplicate
-- detection (phase 1) and French full-text search (phase 5).
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
