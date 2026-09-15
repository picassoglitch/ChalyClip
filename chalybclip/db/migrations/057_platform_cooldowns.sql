-- Per-tenant per-platform publish cooldown — the abuse/rate-limit backoff.
--
-- When a platform fails a publish with a `user_abuse` signal (rate limit,
-- velocity cap, "wait N minutes", daily upload limit), we park that platform
-- until `until`. The rulebook scheduler skips a platform in cooldown so the
-- engine stops feeding a platform that's actively flagging us — which is what
-- gets accounts ghosted/throttled. Self-clears once `until` passes.
CREATE TABLE IF NOT EXISTS platform_cooldowns (
    tenant_id   TEXT NOT NULL,
    platform    TEXT NOT NULL,
    until       TEXT NOT NULL,
    reason      TEXT,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (tenant_id, platform),
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);
