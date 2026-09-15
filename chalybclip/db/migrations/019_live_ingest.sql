-- Live RTMP ingest — phase L.1.
--
-- Phase L architecture is documented at docs/phase_L_live.md.
-- L.1 deliverables: MediaMTX accepts RTMP push -> ChalybClip stores the
-- recording -> operator can run the existing VOD pipeline against it.
--
-- This migration only adds the persistence pieces. No live runner yet
-- (that's L.2+).

-- 1. New flags on `streams` so the existing UI can tell live recordings
-- apart from uploaded VODs without a separate table. Existing rows get
-- 0 / NULL defaults so no migration backfill is needed.
--
-- The status column already exists (TEXT). After L.1 it gains two new
-- accepted values:
--   'live'        — RTMP push currently active
--   'live_ended'  — recording finished, in 24h retention window
-- Old statuses ('ingested', 'running', 'done', 'failed') keep working.

ALTER TABLE streams ADD COLUMN is_live INTEGER NOT NULL DEFAULT 0;
ALTER TABLE streams ADD COLUMN live_started_at TEXT;
ALTER TABLE streams ADD COLUMN live_ended_at TEXT;

-- 2. Per-tenant RTMP stream keys. One active key per tenant at a time;
-- rotation creates a new row + flips revoked_at on the old one. The
-- key value is the secret OBS will use in its push URL, so it MUST be
-- random + long. Generated via secrets.token_urlsafe(32) -> ULID-prefixed
-- string in the repo helper.
--
-- last_used_at is updated on every successful authorize webhook hit
-- so the operator can see "this key was used 12s ago" on the live
-- dashboard.
CREATE TABLE IF NOT EXISTS live_stream_keys (
    id            TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    key_value     TEXT NOT NULL UNIQUE,
    created_at    TEXT NOT NULL,
    revoked_at    TEXT,
    last_used_at  TEXT,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);

-- Partial index for the dashboard's "find this tenant's active key"
-- query. SQLite supports partial indexes since 3.8 (we're on 3.36+).
CREATE INDEX IF NOT EXISTS idx_live_stream_keys_active
    ON live_stream_keys(tenant_id)
    WHERE revoked_at IS NULL;
