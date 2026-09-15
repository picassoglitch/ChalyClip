-- Broadcast send log — the per-tenant daily-broadcast guardrail (Hub
-- phase 10).
--
-- A broadcast is a mass DM: a mistake is irreversible spam. The growth
-- layer caps sends to CHALYBCLIP_HUB_MAX_BROADCASTS_PER_DAY (default 1)
-- per tenant per UTC day. Zernio has no such cap, so it MUST live
-- locally. One row per send (not per draft) — drafts and scheduling
-- don't count against the cap, only the actual fire does.

CREATE TABLE IF NOT EXISTS zernio_broadcast_log (
    id           TEXT PRIMARY KEY,   -- bcl_<ulid>
    tenant_id    TEXT NOT NULL,
    broadcast_id TEXT NOT NULL,      -- Zernio broadcast id
    day          TEXT NOT NULL,      -- YYYY-MM-DD (UTC)
    sent_at      TEXT NOT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_broadcast_log_tenant_day
    ON zernio_broadcast_log (tenant_id, day);
