-- User-level editor preferences on the brand kit (slice H.1).
--
-- The clip editor's right panel previously persisted form values per-clip
-- only. Re-opening any clip (or opening a new one) reset the URL, banner
-- toggle, caption position/font/animation/lead-time — operators had to
-- re-set everything every time. These are *user setup* values, not
-- per-page state.
--
-- This migration adds the missing fields to the tenant's brand_kit so a
-- save in one clip becomes the default for every subsequent clip. The
-- per-clip `clips.overlay_config_json` STILL wins when set — it's the
-- per-clip override; the brand_kit holds the user-level default.
--
-- The four caption knobs (position / font_size / animation / lead_ms)
-- fit inside the existing `caption_style_json` JSON blob — no schema
-- change for those, just an extended JSON shape. This file only adds
-- the three non-JSON-friendly toggle/default columns.

ALTER TABLE brand_kits ADD COLUMN default_platform           TEXT;
ALTER TABLE brand_kits ADD COLUMN banner_enabled_default     INTEGER NOT NULL DEFAULT 0;
ALTER TABLE brand_kits ADD COLUMN banner_show_context_default  INTEGER NOT NULL DEFAULT 0;
ALTER TABLE brand_kits ADD COLUMN banner_show_safezones_default INTEGER NOT NULL DEFAULT 0;
