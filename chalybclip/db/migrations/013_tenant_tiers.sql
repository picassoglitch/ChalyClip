-- Tenant tiers + ChalybClip credit toggle — slice O.1.
--
-- Per-tenant subscription tier drives:
--   * watermark presence on exported clips (free tier always
--     gets the "chalybclip.com" credit burned bottom-right; paid
--     tiers can toggle it off via brand_kits.show_chalybclip_credit)
--   * (future J.2) token allowances, batch limits, billing
--
-- Existing tenants land on `free` so no behavior changes until
-- they're explicitly upgraded.

ALTER TABLE tenants ADD COLUMN tier TEXT NOT NULL DEFAULT 'free';

-- Pro+ tiers can turn off the bottom-right "chalybclip.com" credit.
-- Free tier always gets it (the watermark function ignores this
-- flag when tier='free'). Default ON for everyone, so newly-
-- upgraded tenants keep crediting until they explicitly opt out.
ALTER TABLE brand_kits ADD COLUMN show_nexoclip_credit INTEGER NOT NULL DEFAULT 1;
