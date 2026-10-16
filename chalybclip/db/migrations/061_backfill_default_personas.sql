-- Backfill: every account without a persona gets the `default` one SSO
-- auto-provisioning seeds (same name, languages, prompt and tag as
-- integrations/chalyb/sso_tenant.py). Before 060 that seed only worked for
-- the first account. Idempotent: an account that has any persona is left
-- alone. Portable SQL (SQLite and Postgres).

INSERT INTO personas (id, tenant_id, name, primary_language, target_languages_json,
                      voice_prompt, routing_tags_json, created_at)
SELECT 'default', t.id, 'Default', 'en', '["en", "es"]',
       'You are writing short-form social media captions for a content creator. Write tight, punchy captions that earn the swipe. Lead with a hook — no soft openers. Avoid corporate language and emoji stuffing. Match the energy of the source clip. Output the caption and nothing else.',
       '["general"]', t.created_at
FROM tenants t
WHERE NOT EXISTS (SELECT 1 FROM personas p WHERE p.tenant_id = t.id);
