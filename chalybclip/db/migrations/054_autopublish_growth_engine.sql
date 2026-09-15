-- Growth Engine knobs on the per-tenant auto-publish settings (Phases 2 + 5).
--
--   growth_engine     -- master switch: when 1, hands-free runs the Growth
--                        Engine (per-platform rulebook + Growth Score + fatigue
--                        spacing + allocation) instead of the flat interval drip.
--   growth_min_score  -- publish floor (0-100). A clip whose per-platform Growth
--                        Score is below this is held off that platform (Phase 5
--                        "don't publish / archive").
--   daily_clip_budget -- "how many clips today" (Phase 2 Growth Publish). NULL /
--                        0 = no pool cap; only the per-platform daily ceilings
--                        bound the day.
ALTER TABLE autopublish_settings
    ADD COLUMN growth_engine INTEGER NOT NULL DEFAULT 0;
ALTER TABLE autopublish_settings
    ADD COLUMN growth_min_score INTEGER NOT NULL DEFAULT 40;
ALTER TABLE autopublish_settings
    ADD COLUMN daily_clip_budget INTEGER;
