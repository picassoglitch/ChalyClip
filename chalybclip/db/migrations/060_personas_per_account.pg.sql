-- Postgres version of 060_personas_per_account.sql (same intent — read that
-- file's header). Postgres can change keys in place, so: drop variants' FK
-- to personas and the personas primary key (whatever their names are), copy
-- personas that other accounts use, then key personas by (tenant_id, id) and
-- point variants at it with a composite foreign key. Runs as one implicit
-- transaction (single simple-protocol script).

DO $$
DECLARE r record;
BEGIN
    FOR r IN
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'variants'::regclass AND contype = 'f'
          AND confrelid = 'personas'::regclass
    LOOP
        EXECUTE format('ALTER TABLE variants DROP CONSTRAINT %I', r.conname);
    END LOOP;
    FOR r IN
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'personas'::regclass AND contype = 'p'
    LOOP
        EXECUTE format('ALTER TABLE personas DROP CONSTRAINT %I', r.conname);
    END LOOP;
END $$;

INSERT INTO personas (id, tenant_id, name, primary_language, target_languages_json,
                      voice_prompt, routing_tags_json, created_at)
SELECT DISTINCT p.id, v.tenant_id, p.name, p.primary_language, p.target_languages_json,
       p.voice_prompt, p.routing_tags_json, p.created_at
FROM variants v
JOIN personas p ON p.id = v.persona_id
WHERE NOT EXISTS (
    SELECT 1 FROM personas n WHERE n.tenant_id = v.tenant_id AND n.id = v.persona_id
);

INSERT INTO personas (id, tenant_id, name, primary_language, target_languages_json,
                      voice_prompt, routing_tags_json, created_at)
SELECT DISTINCT p.id, w.tenant_id, p.name, p.primary_language, p.target_languages_json,
       p.voice_prompt, p.routing_tags_json, p.created_at
FROM channel_watches w
JOIN personas p ON p.id = w.persona_id
WHERE NOT EXISTS (
    SELECT 1 FROM personas n WHERE n.tenant_id = w.tenant_id AND n.id = w.persona_id
);

ALTER TABLE personas ADD PRIMARY KEY (tenant_id, id);
ALTER TABLE variants ADD CONSTRAINT variants_tenant_persona_fkey
    FOREIGN KEY (tenant_id, persona_id) REFERENCES personas (tenant_id, id) ON DELETE CASCADE;
