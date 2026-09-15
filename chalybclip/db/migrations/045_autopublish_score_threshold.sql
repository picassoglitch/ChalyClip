-- Auto-publish hands-free score threshold (Fase 2).
--
-- In hands_free mode the pipeline auto-publishes every clip whose viral
-- score is >= this threshold (0.0–1.0). Default 0.6. on_approve mode
-- ignores it (the operator's approve is the gate there).
ALTER TABLE autopublish_settings
    ADD COLUMN score_threshold REAL NOT NULL DEFAULT 0.6;
