-- Zernio publish records — one row per POST /posts we fired.
--
-- WHY LOCAL: Zernio's GET /posts is scoped to the company API key, not
-- to a tenant — rendering it directly would show every tenant's posts
-- to everyone (tenant-isolation violation, CLAUDE rule #1). We record
-- each publish here at create time, keyed by Zernio's post id, and the
-- dashboard history reads THIS table (tenant-filtered), joining live
-- status from Zernio by post id.
--
-- Also powers the publish UX: a clip with a publish record flips to
-- status='published' and leaves the "ready to publish" grid; the
-- history row's View button opens the per-post job page.

CREATE TABLE IF NOT EXISTS zernio_publishes (
    post_id     TEXT PRIMARY KEY,       -- Zernio post _id (job handle)
    tenant_id   TEXT NOT NULL,
    clip_id     TEXT NOT NULL,
    platforms   TEXT NOT NULL,          -- csv of platform ids (tiktok,instagram)
    content     TEXT,                   -- the caption we sent
    created_at  TEXT NOT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_zernio_publishes_tenant
    ON zernio_publishes (tenant_id, created_at DESC);
