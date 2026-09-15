-- Publish options snapshot on the local publish record (Hub phase 4).
--
-- Drafts ("Guardar como borrador") re-publish later from the LOCAL
-- record: the original signed media URL has expired by then, and the
-- per-platform extras (caption overrides, YouTube title, first
-- comment, TikTok privacy) only existed in the original form post.
-- This column snapshots those options as JSON at create time so
-- "Publicar ahora" / "Programar" on a draft rebuilds the exact same
-- createPost payload. NULL for rows that predate this migration and
-- for publishes without extras.

ALTER TABLE zernio_publishes ADD COLUMN options_json TEXT;
