-- Consumption contract (chalyb docs/engines/consumption-contract.md):
-- durable usage delivery, admitted-job bookkeeping, per-stream storage.
--
-- usage_outbox — every usage event (llm.tokens, transcription.seconds,
-- compute.seconds, engine.base) and every reservation settle is written
-- here first, then drained to the hub in batches with backoff. Replaces the
-- fire-and-forget reporter, which dropped events on restart / scale-to-zero.
-- `id` is the event's source_id (the hub dedupes on (engine, source_id), so
-- re-sending is safe). No FK to tenants: a billing row must outlive a
-- deleted tenant until it's delivered.
CREATE TABLE IF NOT EXISTS usage_outbox (
    id              TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    endpoint        TEXT NOT NULL DEFAULT 'usage',
    payload_json    TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    last_error      TEXT,
    created_at      TEXT NOT NULL,
    sent_at         TEXT
);
CREATE INDEX IF NOT EXISTS usage_outbox_due_idx
    ON usage_outbox (status, next_attempt_at);

-- usage_jobs — one row per admitted pipeline run. `id` is the
-- external_job_id sent to /usage/admit (re-admitting the same id updates
-- the same reservation). `kickoff_json` carries the PipelineKickoff for a
-- boost-lane run: the one-shot Cloud Run Job only receives this id and
-- loads everything else from here.
CREATE TABLE IF NOT EXISTS usage_jobs (
    id              TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL,
    stream_id       TEXT NOT NULL,
    operation       TEXT NOT NULL,
    job_class       TEXT NOT NULL DEFAULT 'job',
    reservation_id  TEXT,
    lane            TEXT NOT NULL DEFAULT 'standard',
    boost           INTEGER,
    upload_mb       DOUBLE PRECISION NOT NULL DEFAULT 0,
    source_minutes  DOUBLE PRECISION NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'admitted',
    kickoff_json    TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_jobs_stream_idx ON usage_jobs (stream_id);

-- What a stream's artifacts occupy, measured at the end of a successful run
-- (after the source reclaim). SUM over a tenant is the cheap storage figure
-- admission sends as `storage_mb_after`. BIGINT: byte counts overflow a
-- Postgres INTEGER.
ALTER TABLE streams ADD COLUMN storage_bytes BIGINT NOT NULL DEFAULT 0;
