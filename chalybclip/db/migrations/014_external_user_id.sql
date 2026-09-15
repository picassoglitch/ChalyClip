-- Chalyb integration — slice NX.1.
--
-- Adds `tenants.external_user_id` so we can detect duplicates when Chalyb
-- calls POST /api/admin/tenants. The Chalyb side uses its Supabase Auth
-- user_id as the stable cross-system identifier; ChalybClip stores that on
-- the tenant row so a re-provision call (admin retry, webhook redelivery)
-- maps back to the same tenant instead of creating a second one.
--
-- Why UNIQUE WHERE NOT NULL: existing tenants created via the CLI before
-- this slice landed have no Chalyb identity. Allowing NULL keeps them valid
-- without backfill, while the partial unique index still forbids two tenants
-- from claiming the same external id.
--
-- See docs/chalyb_integration.md for the full contract.

ALTER TABLE tenants ADD COLUMN external_user_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_tenants_external_user_id
  ON tenants(external_user_id)
  WHERE external_user_id IS NOT NULL;
