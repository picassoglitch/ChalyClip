-- Zernio webhook event log + live publish status (Hub phase 2).
--
-- zernio_events: one row per inbound webhook delivery, keyed by
-- Zernio's stable event id (payload.id). Delivery is at-least-once,
-- so the PRIMARY KEY is the dedup: a redelivery's INSERT OR IGNORE
-- no-ops and the receiver ACKs 200 without reprocessing. The raw
-- payload is kept verbatim for replay/debugging and for the later
-- phases (calendar, inbox) that will backfill their stores from it.
-- tenant_id is resolved from the payload's profileId (or the post id
-- via zernio_publishes) by the background processor; NULL when the
-- event doesn't map to a known tenant (kept anyway — never drop data
-- on a resolution miss).

CREATE TABLE IF NOT EXISTS zernio_events (
    event_id     TEXT PRIMARY KEY,      -- Zernio payload.id (dedup key)
    type         TEXT NOT NULL,         -- e.g. post.published, comment.received
    payload      TEXT NOT NULL,         -- raw JSON body, verbatim
    profile_id   TEXT,                  -- Zernio profileId when present
    tenant_id    TEXT,                  -- resolved tenant, NULL if unknown
    received_at  TEXT NOT NULL,
    processed    INTEGER NOT NULL DEFAULT 0,
    processed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_zernio_events_tenant
    ON zernio_events (tenant_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_zernio_events_unprocessed
    ON zernio_events (processed, received_at);

-- Live status on the local publish record (fed by post.* webhooks so
-- the dashboard stops polling Zernio for history rows). status follows
-- Zernio's vocab (scheduled/publishing/published/failed/partial/
-- cancelled); platforms_json is the per-platform result array from the
-- last event, verbatim.
ALTER TABLE zernio_publishes ADD COLUMN status TEXT;
ALTER TABLE zernio_publishes ADD COLUMN platforms_json TEXT;
ALTER TABLE zernio_publishes ADD COLUMN updated_at TEXT;
