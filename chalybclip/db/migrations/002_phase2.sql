-- Phase 2 schema, lock-down version. Any further change is migration 003+.
--
-- Adds the storage shape for:
--   * webhook subscriptions (Task 12)
--   * OAuth refresh tokens on connected_accounts (Tasks 8-10)
--   * native-publisher metadata on publish_jobs (Tasks 8-9)
--   * vision-LLM rescore verdicts on candidates (Tasks 3-4)
--   * per-tenant budget + quota knobs the governor enforces (Task 1)
--
-- All ALTER TABLEs add columns at the end; SQLite supports this without
-- rewriting the table. NULL defaults across the board so existing rows
-- stay valid; the governor treats NULL budget/quota as "unlimited".

-- ---------- Webhook subscriptions ----------

CREATE TABLE webhook_subscriptions (
    id                TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    url               TEXT NOT NULL,
    types_json        TEXT NOT NULL DEFAULT '[]',  -- array of event type strings
    secret            TEXT NOT NULL,                -- HMAC-SHA256 secret, 32+ bytes hex
    status            TEXT NOT NULL DEFAULT 'active',  -- active | disabled
    created_at        TEXT NOT NULL,
    last_dispatch_ts  TEXT,                         -- newest event ts already delivered
    failure_count     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_webhook_subscriptions_tenant ON webhook_subscriptions (tenant_id, status);

-- ---------- Connected accounts: OAuth + status ----------

ALTER TABLE connected_accounts ADD COLUMN refresh_token TEXT;
ALTER TABLE connected_accounts ADD COLUMN expires_at    TEXT;
ALTER TABLE connected_accounts ADD COLUMN scopes_json   TEXT;
-- 'active' | 'auth_failed' | 'disabled'
ALTER TABLE connected_accounts ADD COLUMN status        TEXT NOT NULL DEFAULT 'active';
CREATE INDEX idx_connected_accounts_tenant_status ON connected_accounts (tenant_id, status);

-- ---------- Publish jobs: native-publisher metadata ----------

ALTER TABLE publish_jobs ADD COLUMN external_url           TEXT;
ALTER TABLE publish_jobs ADD COLUMN platform_metadata_json TEXT;

-- ---------- Candidates: vision-LLM rescore verdict ----------

-- Set by Task 3's vision_rescore service when --vision-rescore is on. Until
-- then they stay NULL, and detector ordering falls back to `score`.
ALTER TABLE candidates ADD COLUMN rescore_score  REAL;
ALTER TABLE candidates ADD COLUMN rescore_reason TEXT;
ALTER TABLE candidates ADD COLUMN rescore_model  TEXT;

-- ---------- Tenants: budget governor knobs ----------

-- NULL == unlimited. Governor (Task 1) reads these per-call and refuses
-- LLM / publish work that would push today's totals past the ceiling.
ALTER TABLE tenants ADD COLUMN daily_llm_budget_usd_micros INTEGER;
ALTER TABLE tenants ADD COLUMN daily_publish_limit         INTEGER;
ALTER TABLE tenants ADD COLUMN rescore_concurrency_cap     INTEGER NOT NULL DEFAULT 4;
