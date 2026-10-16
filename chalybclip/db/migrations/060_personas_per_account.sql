-- Persona ids are per account (tenant), not global.
--
-- personas.id used to be the table's PRIMARY KEY, so two accounts could
-- never both have a persona called "default": the persona seeded on SSO
-- auto-provisioning only stuck for the first account, every later account
-- had none (URL jobs refused with "make a persona first"), and
-- PersonasRepo.upsert's ON CONFLICT(id) could overwrite another account's
-- persona. The key becomes (tenant_id, id) and variants point at it with a
-- composite foreign key.
--
-- Before the key changes, every (account, persona id) still in use without
-- its own row — variants or channel watches pointing at a persona another
-- account owns — gets a copy of that persona for the using account, so no
-- reference is lost.
--
-- SQLite can't change a primary key in place: rebuild both tables with
-- foreign keys off (so DROP TABLE doesn't cascade). The Postgres version of
-- this migration is 060_personas_per_account.pg.sql.

PRAGMA foreign_keys = OFF;
BEGIN;

CREATE TABLE personas_new (
    id                    TEXT NOT NULL,
    tenant_id             TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name                  TEXT NOT NULL,
    primary_language      TEXT NOT NULL,
    target_languages_json TEXT NOT NULL DEFAULT '[]',
    voice_prompt          TEXT NOT NULL,
    routing_tags_json     TEXT NOT NULL DEFAULT '[]',
    created_at            TEXT NOT NULL,
    PRIMARY KEY (tenant_id, id)
);

INSERT INTO personas_new (id, tenant_id, name, primary_language, target_languages_json,
                          voice_prompt, routing_tags_json, created_at)
SELECT id, tenant_id, name, primary_language, target_languages_json,
       voice_prompt, routing_tags_json, created_at
FROM personas;

INSERT INTO personas_new (id, tenant_id, name, primary_language, target_languages_json,
                          voice_prompt, routing_tags_json, created_at)
SELECT DISTINCT p.id, v.tenant_id, p.name, p.primary_language, p.target_languages_json,
       p.voice_prompt, p.routing_tags_json, p.created_at
FROM variants v
JOIN personas p ON p.id = v.persona_id
WHERE NOT EXISTS (
    SELECT 1 FROM personas_new n WHERE n.tenant_id = v.tenant_id AND n.id = v.persona_id
);

INSERT INTO personas_new (id, tenant_id, name, primary_language, target_languages_json,
                          voice_prompt, routing_tags_json, created_at)
SELECT DISTINCT p.id, w.tenant_id, p.name, p.primary_language, p.target_languages_json,
       p.voice_prompt, p.routing_tags_json, p.created_at
FROM channel_watches w
JOIN personas p ON p.id = w.persona_id
WHERE NOT EXISTS (
    SELECT 1 FROM personas_new n WHERE n.tenant_id = w.tenant_id AND n.id = w.persona_id
);

CREATE TABLE variants_new (
    id               TEXT PRIMARY KEY,
    clip_id          TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    tenant_id        TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    persona_id       TEXT NOT NULL,
    language         TEXT NOT NULL,
    caption          TEXT NOT NULL,
    title_card_text  TEXT NOT NULL DEFAULT '',
    hashtags_json    TEXT NOT NULL DEFAULT '[]',
    model            TEXT,
    created_at       TEXT NOT NULL,
    FOREIGN KEY (tenant_id, persona_id) REFERENCES personas(tenant_id, id) ON DELETE CASCADE
);

INSERT INTO variants_new (id, clip_id, tenant_id, persona_id, language, caption,
                          title_card_text, hashtags_json, model, created_at)
SELECT id, clip_id, tenant_id, persona_id, language, caption,
       title_card_text, hashtags_json, model, created_at
FROM variants;

DROP TABLE variants;
DROP TABLE personas;
ALTER TABLE personas_new RENAME TO personas;
ALTER TABLE variants_new RENAME TO variants;

CREATE INDEX idx_personas_tenant ON personas (tenant_id);
CREATE INDEX idx_variants_tenant_created ON variants (tenant_id, created_at);
CREATE INDEX idx_variants_clip ON variants (clip_id);

COMMIT;
PRAGMA foreign_keys = ON;
