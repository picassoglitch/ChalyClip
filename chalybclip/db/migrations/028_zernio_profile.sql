-- Zernio integration — tenant ↔ profileId mapping.
--
-- Replaces the upload-post.com integration (migration 022). Each
-- ChalybClip tenant maps to ONE Zernio `profileId` — a free-form
-- namespacing string we derive from the tenant id and persist here on
-- the first connect/publish click. Zernio scopes connected social
-- accounts + posts by this value (passed as `profileId` on the
-- /connect and /accounts calls).
--
-- Column is nullable: tenants that haven't started a publish flow yet
-- have NULL. The ensure_profile_for_tenant() helper backfills on first
-- use. The legacy `upload_post_profile_username` column (migration 022)
-- is left in place but dormant — no app code reads or writes it now.

ALTER TABLE tenants ADD COLUMN zernio_profile_id TEXT;

-- No index needed — the column is only read via the tenant's own row
-- lookup (already indexed by id).
