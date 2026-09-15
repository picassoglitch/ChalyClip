-- Auto-publish ("Piloto automático") settings, per-tenant (Publish Center tab).
--
-- When enabled, ChalybClip auto-enqueues clips to Zernio with their burned-in
-- render (hooks / captions already composited) — the operator no longer has
-- to pick platforms + post by hand. Routed entirely through Zernio; the
-- legacy publish_jobs worker is gone.
--
-- mode:
--   on_approve  -- auto-publish when the operator approves a clip ("Ship")
--   hands_free  -- (phase 2) auto-publish every generated clip, no review
-- post_mode is the Zernio publish mode: 'queue' uses the recurring slots from
--   the Programación tab (spaced out, policy-safe); 'now' fires immediately.
-- targets is a csv of Zernio platform ids (e.g. "tiktok,instagram").
-- daily_cap is an anti-spam ceiling counted per UTC day (0 = unlimited).
CREATE TABLE IF NOT EXISTS autopublish_settings (
    tenant_id   TEXT PRIMARY KEY,
    enabled     INTEGER NOT NULL DEFAULT 0,
    mode        TEXT NOT NULL DEFAULT 'on_approve',
    targets     TEXT,
    post_mode   TEXT NOT NULL DEFAULT 'queue',
    daily_cap   INTEGER NOT NULL DEFAULT 10,
    updated_at  TEXT NOT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);
