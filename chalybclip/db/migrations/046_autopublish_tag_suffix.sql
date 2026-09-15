-- Auto-publish fixed tag/handle suffix (Publish Center "Piloto automático").
--
-- A per-tenant free-text line appended to every auto-published / auto-
-- programmed caption: the creator's @handles + brand hashtags they want on
-- every post (e.g. "@minombre · seguime en Kick · #gaming #clips"). The AI
-- per-clip hashtags still come from the clip's variant; this is the fixed
-- suffix that rides on top. Empty string = nothing appended.
ALTER TABLE autopublish_settings
    ADD COLUMN tag_suffix TEXT NOT NULL DEFAULT '';
