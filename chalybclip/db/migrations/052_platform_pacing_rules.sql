-- Per-tenant per-platform publishing rulebook (Growth Engine, Phase 1).
--
-- The hard rules that protect account health: a daily ceiling, a minimum gap
-- between posts, caption-length + hashtag-count bands, and a randomized delay
-- window. Rows here are OVERRIDES — any platform a tenant hasn't tuned falls
-- back to chalybclip.publish.pacing.DEFAULT_PLATFORM_RULES in code, so an empty
-- table still yields the shipped conservative defaults.
--
-- platform is the canonical lowercased id (twitter, not "x"); a day is a UTC
-- calendar day, matching the existing cap accounting in hub_publish_jobs.
CREATE TABLE IF NOT EXISTS platform_pacing_rules (
    tenant_id          TEXT NOT NULL,
    platform           TEXT NOT NULL,
    max_per_day        INTEGER NOT NULL,
    min_gap_minutes    INTEGER NOT NULL,
    caption_min_chars  INTEGER NOT NULL DEFAULT 0,
    caption_max_chars  INTEGER NOT NULL DEFAULT 2200,
    hashtag_min        INTEGER NOT NULL DEFAULT 0,
    hashtag_max        INTEGER NOT NULL DEFAULT 30,
    jitter_minutes     INTEGER NOT NULL DEFAULT 0,
    caption_style      TEXT NOT NULL DEFAULT 'default',
    enabled            INTEGER NOT NULL DEFAULT 1,
    updated_at         TEXT NOT NULL,
    PRIMARY KEY (tenant_id, platform),
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
);
