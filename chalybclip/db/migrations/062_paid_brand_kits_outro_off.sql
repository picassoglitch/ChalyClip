-- Paid brand kits stop appending the Chalyb end card by default (owner,
-- 2026-11-07). New paid kits are created with show_chalybclip_outro = 0; this
-- brings existing paid kits in line. A paid user can switch it back on in the
-- kit form. Free kits are untouched: the renderer always adds the end card
-- and watermark for free tenants whatever the flag says.
--
-- "Paid" mirrors chalybclip/tiers.py: the canonical pro / all_access tiers
-- plus their aliases. Unknown labels read as free there, so they are left
-- alone here too.
--
-- Idempotent: only rows still at 1 are touched; a second run changes nothing.
-- Portable SQL (SQLite and Postgres).
UPDATE brand_kits
SET show_chalybclip_outro = 0
WHERE show_chalybclip_outro <> 0
  AND tenant_id IN (
    SELECT id FROM tenants
    WHERE LOWER(TRIM(tier)) IN (
      'pro', 'all_access',
      'partner', 'partners', 'enterprise', 'allaccess', 'all-access', 'vip'
    )
  );
