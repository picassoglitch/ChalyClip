-- Clip trim (auto-trim around integrity issues) — slice G.4b.
--
-- The G.4 integrity detector flags freeze frames + silent gaps.
-- G.4b adds a one-click "auto-trim around the issues" button that
-- shrinks the clip to its longest clean window. Revert undoes that
-- and restores the original bounds.
--
-- We track the originals on the clip row so revert is a single
-- query — no separate audit table needed. NULL = clip has never been
-- trimmed; revert is a no-op.

ALTER TABLE clips ADD COLUMN original_start_s REAL;
ALTER TABLE clips ADD COLUMN original_end_s REAL;
