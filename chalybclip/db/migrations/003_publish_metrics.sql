-- Phase 3 schema migration 003 - publish_metrics.
--
-- One row per (publish_job, fetch). The ingest worker writes a fresh row
-- on each poll cycle; the dashboard renders the latest row per clip and
-- the calibration loop reads the time series. No UPDATEs - a row is the
-- snapshot at fetch time.
--
-- Normalized stats across platforms (some won't expose every column):
--
--   views            integer count of plays
--   likes            integer
--   comments         integer
--   shares           integer
--   retention_pct    0.0-1.0  avg_view_duration / clip_duration
--   ctr              0.0-1.0  if the platform exposes click-through rate
--   raw_metadata_json  the full API response - kept for debugging /
--                      future-proofing when we want a metric we didn't
--                      think to extract today.
--
-- Foreign keys ON DELETE CASCADE so deleting a tenant's data wipes the
-- whole metric history without orphans.

CREATE TABLE publish_metrics (
    id                  TEXT PRIMARY KEY,
    tenant_id           TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    publish_job_id      TEXT NOT NULL REFERENCES publish_jobs(id) ON DELETE CASCADE,
    platform            TEXT NOT NULL,
    fetched_at          TEXT NOT NULL,
    views               INTEGER,
    likes               INTEGER,
    comments            INTEGER,
    shares              INTEGER,
    retention_pct       REAL,
    ctr                 REAL,
    raw_metadata_json   TEXT,
    created_at          TEXT NOT NULL
);

-- The dashboard's per-clip outcome card asks for the latest reading on
-- one job; the calibration loop scans many jobs ordered by fetched_at.
-- Composite index covers both.
CREATE INDEX idx_publish_metrics_tenant_job_fetched
    ON publish_metrics (tenant_id, publish_job_id, fetched_at);

-- Cross-job calibration scans ("all sent jobs in the last 7 days for
-- this platform") want this shape.
CREATE INDEX idx_publish_metrics_tenant_platform_fetched
    ON publish_metrics (tenant_id, platform, fetched_at);
