-- Auto-retry ledger (Hub phase 6).
--
-- On a post.failed webhook with a TRANSIENT error class, the hub fires
-- ONE automatic retry after a delay, then stops. This table is the
-- once-only guard: a row is claimed (INSERT OR IGNORE) BEFORE the
-- retry is scheduled, so an at-least-once redelivery of post.failed
-- can't schedule a second retry. Keyed by Zernio post id — works even
-- for posts that aren't in zernio_publishes / hub_publish_jobs.

CREATE TABLE IF NOT EXISTS zernio_auto_retries (
    post_id      TEXT PRIMARY KEY,   -- Zernio post _id
    tenant_id    TEXT,               -- resolved tenant, NULL if unknown
    attempted_at TEXT NOT NULL,
    outcome      TEXT                -- 'scheduled' | 'ok' | 'failed'
);
