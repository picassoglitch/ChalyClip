-- Phase 3 schema migration 004 - webhook_secret_versions.
--
-- Tracks the history of past webhook secrets during their grace window so
-- subscribers can rotate without dropping in-flight deliveries. The
-- `webhook_subscriptions.secret` column always holds the *current* secret
-- (what we sign with); rows in this table are PRIOR secrets that the
-- subscriber may still verify against until `expires_at`.
--
-- Rotation flow:
--   1. mint a new secret
--   2. write the OLD secret to this table with expires_at = now + ttl_s
--   3. update webhook_subscriptions.secret = new
--
-- The dispatcher signs with the current secret only. Subscribers can list
-- their active secrets via GET /webhooks/{id}/secrets and try each in turn.

CREATE TABLE webhook_secret_versions (
    id              TEXT PRIMARY KEY,
    subscription_id TEXT NOT NULL REFERENCES webhook_subscriptions(id) ON DELETE CASCADE,
    tenant_id       TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    secret          TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX idx_webhook_secret_versions_sub
    ON webhook_secret_versions (tenant_id, subscription_id, expires_at);
