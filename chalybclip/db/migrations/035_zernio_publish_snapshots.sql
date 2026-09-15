-- Per-post daily metric snapshots (Hub phase 7).
--
-- One row per (post, UTC day): the post's metrics as of that day,
-- captured by the snapshot job. PERSIST-ONLY — this is the substrate
-- for a future clip-selection feedback loop (which VOD moments produce
-- winning clips); there is NO ML or scoring here yet, just history.
--
-- metrics_json is the normalized metric dict (no fake zeros — absent
-- metrics are absent, not 0). UNIQUE (post_id, day) makes the daily
-- snapshot idempotent: re-running the job on the same day updates the
-- row instead of duplicating it.

CREATE TABLE IF NOT EXISTS zernio_publish_snapshots (
    id            TEXT PRIMARY KEY,      -- snp_<ulid>
    tenant_id     TEXT NOT NULL,
    post_id       TEXT NOT NULL,         -- Zernio (or external) post id
    day           TEXT NOT NULL,         -- YYYY-MM-DD (UTC)
    metrics_json  TEXT NOT NULL,         -- normalized post-level metrics
    platforms_json TEXT,                 -- per-platform metrics, verbatim
    captured_at   TEXT NOT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_post_day
    ON zernio_publish_snapshots (post_id, day);
CREATE INDEX IF NOT EXISTS idx_snapshots_tenant
    ON zernio_publish_snapshots (tenant_id, day DESC);
