-- Hub publish jobs — the internal service API's job records (phase 3).
--
-- One row per /api/internal/v1/publish (or per clip in /batch) fired by
-- ChalybOBS / Chalyb / ChalybClip itself. Differs from zernio_publishes
-- (dashboard publishes of LOCAL clips) in that the media is an external
-- URL and the caller addresses tenants by id with a service token.
--
-- Idempotency: UNIQUE (tenant_id, idempotency_key) — a replayed key
-- returns the original job instead of double-posting. Live status is
-- fed by the phase-2 webhook processor via zernio_post_id.

CREATE TABLE IF NOT EXISTS hub_publish_jobs (
    job_id          TEXT PRIMARY KEY,        -- hpj_<ulid>
    tenant_id       TEXT NOT NULL,
    idempotency_key TEXT,                    -- caller-supplied, nullable
    source          TEXT NOT NULL,           -- chalybobs | chalybclip | chalyb
    mode            TEXT NOT NULL,           -- now | queue | schedule | draft
    targets         TEXT NOT NULL,           -- csv of resolved platforms
    video_url       TEXT NOT NULL,
    title           TEXT,
    caption         TEXT,
    scheduled_for   TEXT,                    -- resolved ISO-8601 (schedule/best-time)
    zernio_post_id  TEXT,                    -- NULL until Zernio accepts the post
    status          TEXT NOT NULL DEFAULT 'pending',
    platforms_json  TEXT,                    -- per-platform results (webhook-fed)
    error           TEXT,                    -- structured error code on failure
    created_at      TEXT NOT NULL,
    updated_at      TEXT,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_hub_jobs_idem
    ON hub_publish_jobs (tenant_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_hub_jobs_tenant
    ON hub_publish_jobs (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_hub_jobs_post
    ON hub_publish_jobs (zernio_post_id);
