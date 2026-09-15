-- Publishing safe trap — per-brand-kit safety config.
--
-- The safe trap computes anti-shadowban posting windows (min spacing,
-- daily cap, quiet hours, jitter) per platform. By default it's advisory:
-- the dashboard surfaces a risk score + recommended time. When
-- `safe_schedule_enabled = 1`, auto-publish instead schedules each job
-- into the next compliant window and the publish drain hard-gates posts
-- that would fire inside a blocked window (deferring them).
--
--   safe_schedule_enabled — opt-in hard gate / auto-schedule mode (else advisory).
--   safety_policy_json     — per-platform overrides, {platform: {field: value}}.
--                            NULL = use the built-in PLATFORM_DEFAULTS.
--   content_timezone       — IANA tz for quiet-hours math (no tenant TZ exists).

ALTER TABLE brand_kits ADD COLUMN safe_schedule_enabled INTEGER NOT NULL DEFAULT 0;
ALTER TABLE brand_kits ADD COLUMN safety_policy_json TEXT;
ALTER TABLE brand_kits ADD COLUMN content_timezone TEXT NOT NULL DEFAULT 'UTC';
