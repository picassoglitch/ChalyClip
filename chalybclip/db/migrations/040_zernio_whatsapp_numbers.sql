-- WhatsApp number provisioning status (Hub phase 12, feature-flagged).
--
-- Fed by the whatsapp.number.* webhooks (activated / declined /
-- action_required / verification_required / suspended / reactivated /
-- released). One row per Zernio social account; the latest status
-- wins. Read only when FEATURE_WHATSAPP is on. Tenant-free at rest
-- (keyed by account_id), resolved to a tenant at read time like the
-- other webhook-fed stores.

CREATE TABLE IF NOT EXISTS zernio_whatsapp_numbers (
    account_id  TEXT PRIMARY KEY,   -- Zernio social account id
    status      TEXT NOT NULL,      -- activated | declined | suspended | ...
    detail      TEXT,               -- optional message from the event
    updated_at  TEXT NOT NULL
);
