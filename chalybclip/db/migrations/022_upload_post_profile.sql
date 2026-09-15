-- upload-post.com integration — tenant ↔ profile mapping.
--
-- Each ChalybClip tenant maps to ONE upload-post "user profile" (their
-- multi-tenant primitive). We create the profile lazily on first
-- connect/publish click, then store the username here so subsequent
-- calls reuse it. The username is opaque from upload-post's POV —
-- we pass it as the `user` field on every upload-post API call.
--
-- Column is nullable: tenants that haven't started a publish flow
-- yet have NULL. The ensure_profile_for_tenant() helper backfills
-- on first use.

ALTER TABLE tenants ADD COLUMN upload_post_profile_username TEXT;

-- No index needed — the column is only read via the tenant's own
-- row lookup (already indexed by id).
