-- Publishability verdict persistence — slice G.2.
--
-- compute_publishability() runs on every clip_detail render but its
-- result was never written back. Two consequences we want to fix:
--   1) Inbox + streams grid can't show a "publish_ready / needs_edit /
--      reject" chip without re-running the scorer per row.
--   2) The verdict is recomputed for every page refresh even though
--      the inputs (breakdown + overlay_config) haven't changed.
--
-- Cache the score + status on the clip row. Recompute happens any
-- time the operator saves overlay_config; downstream surfaces read
-- the cached values.
--
-- score: 0-100 integer (the verdict.score field).
-- status: one of 'publish_ready' / 'needs_edit' / 'reject' (the
-- verdict.status field). NULL means "never scored yet" — surfaces
-- fall back to the on-demand compute in that case so existing rows
-- don't break.

ALTER TABLE clips ADD COLUMN publishability_score INTEGER;
ALTER TABLE clips ADD COLUMN publishability_status TEXT;

CREATE INDEX IF NOT EXISTS idx_clips_publishability_status
  ON clips (tenant_id, publishability_status);
