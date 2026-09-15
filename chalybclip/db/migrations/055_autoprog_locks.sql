-- Per-tenant auto-program lock — idempotency for "Auto-programar todo".
--
-- The bulk scheduler used to guard against a double-click with an IN-MEMORY
-- flag, which doesn't hold across multiple web workers (Railway runs several):
-- two near-simultaneous clicks each planned a full drip series, double-booking
-- the queue. This table is the cross-worker lock — a tenant holds at most one
-- row; a fresh row blocks a second run, a stale one (heartbeat older than the
-- staleness window) is reclaimed so a crashed run never wedges the tenant.
CREATE TABLE IF NOT EXISTS autoprog_locks (
    tenant_id   TEXT PRIMARY KEY,
    token       TEXT NOT NULL,
    claimed_at  TEXT NOT NULL,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);
