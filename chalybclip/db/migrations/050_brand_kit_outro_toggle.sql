-- Pro-tier "show chalybclip end-card outro" toggle.
--
-- Every exported clip ends with a short bundled chalybclip end card
-- (assets/outro.mp4), concatenated after the clip. Free tier ALWAYS
-- gets it; paid tiers can switch it off per brand kit via this flag.
-- Default ON for everyone (mirrors show_chalybclip_credit), so newly-
-- upgraded tenants keep the end card until they explicitly opt out.
--
-- Portable SQL (runs post-baseline on both SQLite and Postgres).
ALTER TABLE brand_kits ADD COLUMN show_nexoclip_outro INTEGER NOT NULL DEFAULT 1;
