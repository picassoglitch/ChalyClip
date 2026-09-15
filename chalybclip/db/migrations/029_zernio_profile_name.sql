-- Zernio profile display name.
--
-- Migration 028 added tenants.zernio_profile_id (the server-generated
-- Zernio profile `_id`). The operator now names the profile when they
-- create it from the publish dashboard (POST /profiles); we store that
-- name here so the dashboard can show "Profile: <name>" without an
-- extra round-trip to Zernio.
--
-- Nullable: cleared together with zernio_profile_id on unlink, and NULL
-- for tenants that haven't created a profile yet.

ALTER TABLE tenants ADD COLUMN zernio_profile_name TEXT;
