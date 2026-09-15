-- Pre-publish Growth Score cache, one row per clip (Growth Engine, Phase 5/6).
--
-- The Growth Score model runs BEFORE publishing and returns an overall growth
-- score, a per-platform fit + publish/skip verdict, optimization tips, and the
-- clip's content tags. We cache the whole card as JSON (card_json) plus the two
-- columns the allocator/fatigue passes query hot:
--   overall_score  -- ranks the day's publish pool ("how many clips today")
--   content_tags   -- json array of theme descriptors; drives fatigue spacing
--
-- Idempotent per clip: recomputing overwrites (PRIMARY KEY clip_id). tenant_id
-- rides along for isolation + the "recently published tags" history query.
CREATE TABLE IF NOT EXISTS clip_growth_scores (
    clip_id        TEXT PRIMARY KEY,
    tenant_id      TEXT NOT NULL,
    overall_score  INTEGER NOT NULL,
    decision       TEXT NOT NULL,
    content_tags   TEXT NOT NULL DEFAULT '[]',
    card_json      TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (clip_id) REFERENCES clips(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_clip_growth_scores_tenant
    ON clip_growth_scores (tenant_id, created_at);
