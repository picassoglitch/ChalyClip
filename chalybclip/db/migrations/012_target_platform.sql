-- Target-platform auto-config — slice K.5.
--
-- Operators pick where they're posting (TikTok / Reels / Shorts /
-- Kick / All) as the FIRST decision in the editor. The system then
-- writes every downstream setting (safe-zone target, fit_mode,
-- caption position/size/animation/lead_ms, banner variant, top hook,
-- Clip Style) to match the platform's conventions.
--
-- This column on brand_kits stores the operator's DEFAULT target so
-- the next clip they open auto-applies the same platform's preset.
-- Per-clip overrides land in clips.overlay_config_json under the
-- `target_platform` key, same pattern as clip_style.

ALTER TABLE brand_kits ADD COLUMN target_platform TEXT;
