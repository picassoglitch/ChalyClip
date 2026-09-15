-- Clip Style presets + Kick banner variants + top hook box (slice I.1).
--
-- "Clip Style" is the keystone abstraction: a single named preset that
-- bundles banner variant + top-hook config + caption preset + animation
-- intensity + safezone behavior. The 5 shipped presets are:
--
--   repost_page_viral  — the default, Kick-style black banner + top white
--                        hook box, karaoke captions, high animation. Looks
--                        like a viral Reels repost page.
--   clean_creator     — minimal URL only, subtle captions, no top hook.
--   gaming_chaos      — heavy emphasis, max animation, larger captions.
--   documentary       — long-form story style, less aggressive.
--   minimal_native    — no banner, no hook, captions only.
--
-- The `bottom_banner_style` field carries the Kick banner variant that's
-- active *inside* whichever clip style is picked:
--   kick_black_bar_classic / kick_green_block / kick_minimal_url /
--   kick_repost_page (the recommended default).
--
-- Like 010_user_prefs.sql these are user-level defaults on the brand_kit.
-- The per-clip `clips.overlay_config_json` STILL wins when explicitly set;
-- brand_kit holds the user-level default.

ALTER TABLE brand_kits ADD COLUMN clip_style                   TEXT;
ALTER TABLE brand_kits ADD COLUMN bottom_banner_style          TEXT;
ALTER TABLE brand_kits ADD COLUMN banner_live_badge_default    INTEGER NOT NULL DEFAULT 0;
ALTER TABLE brand_kits ADD COLUMN top_hook_enabled_default     INTEGER NOT NULL DEFAULT 0;
ALTER TABLE brand_kits ADD COLUMN top_hook_style_default       TEXT;
