"""Persona ids are per account (migrations 060 + 061).

Before 060, personas.id was the table's primary key: only the first account
could have a persona called "default" (the SSO seed silently failed for every
later one), and PersonasRepo.upsert's ON CONFLICT(id) could overwrite another
account's persona.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chalybclip.db import Database, PersonasRepo, TenantsRepo, apply_migrations
from chalybclip.tenancy import bound_tenant

from .._db_backend import INTEGRITY_ERRORS, migrated_database

_MIGRATIONS = Path(__file__).resolve().parents[2] / "chalybclip" / "db" / "migrations"
_T = "2026-01-01T00:00:00+00:00"


async def _persona(db: Database, tenant: str, pid: str, prompt: str = "v") -> None:
    with bound_tenant(tenant):
        await PersonasRepo(db).create(
            persona_id=pid, name=pid, primary_language="es",
            target_languages=["es"], voice_prompt=prompt,
        )


async def test_two_accounts_can_both_have_default(tmp_path: Path) -> None:
    async for db in migrated_database(tmp_path):
        a = (await TenantsRepo(db).create(name="A")).id
        b = (await TenantsRepo(db).create(name="B")).id
        await _persona(db, a, "default", "prompt A")
        await _persona(db, b, "default", "prompt B")
        with bound_tenant(a):
            pa = await PersonasRepo(db).get("default")
        with bound_tenant(b):
            pb = await PersonasRepo(db).get("default")
        assert pa is not None and pa.voice_prompt == "prompt A"
        assert pb is not None and pb.voice_prompt == "prompt B"
        # Same account, same id: still refused.
        with pytest.raises(INTEGRITY_ERRORS):
            await _persona(db, a, "default")


async def test_upsert_never_touches_another_account(tmp_path: Path) -> None:
    async for db in migrated_database(tmp_path):
        a = (await TenantsRepo(db).create(name="A")).id
        b = (await TenantsRepo(db).create(name="B")).id
        await _persona(db, a, "gamer", "A's voice")
        with bound_tenant(b):
            got = await PersonasRepo(db).upsert(
                persona_id="gamer", name="B gamer", primary_language="en",
                target_languages=["en"], voice_prompt="B's voice",
            )
        assert got.tenant_id == b and got.voice_prompt == "B's voice"
        with bound_tenant(a):
            pa = await PersonasRepo(db).get("gamer")
        assert pa is not None and pa.voice_prompt == "A's voice"
        # Upsert within the same account updates in place.
        with bound_tenant(a):
            await PersonasRepo(db).upsert(
                persona_id="gamer", name="A2", primary_language="es",
                target_languages=[], voice_prompt="A2",
            )
            assert (await PersonasRepo(db).get("gamer")).name == "A2"  # type: ignore[union-attr]
            assert len(await PersonasRepo(db).list_for_tenant()) == 1


async def test_migration_keeps_every_reference_and_backfills(tmp_path: Path) -> None:
    """Old data: A owns `default`; B has a variant and C a channel watch that
    point at it (possible before 060); D has no persona at all."""
    old = tmp_path / "old_migrations"
    old.mkdir()
    for f in sorted(_MIGRATIONS.glob("0[0-5][0-9]_*.sql")):
        if not f.name.endswith(".pg.sql"):
            shutil.copy(f, old / f.name)
    db = Database(str(tmp_path / "old.db"))
    try:
        assert await apply_migrations(db, migrations_dir=old) == 59
        conn = await db.connect()
        for t in ("ten_a", "ten_b", "ten_c", "ten_d"):
            await conn.execute(
                "INSERT INTO tenants (id, name, created_at) VALUES (?, ?, ?)", (t, t, _T)
            )
        await conn.execute(
            "INSERT INTO personas (id, tenant_id, name, primary_language, voice_prompt, "
            "created_at) VALUES ('default', 'ten_a', 'A default', 'es', 'A prompt', ?)",
            (_T,),
        )
        await conn.execute(
            "INSERT INTO streams (id, tenant_id, vod_url, platform, duration_s, "
            "source_video_path, source_audio_path, status, created_at) "
            "VALUES ('str_b', 'ten_b', 'https://kick.com/x', 'kick', 1, '/v', '/a', 'done', ?)",
            (_T,),
        )
        await conn.execute(
            "INSERT INTO clips (id, stream_id, tenant_id, start_s, end_s, duration_s, "
            "width, height, path, status, created_at) "
            "VALUES ('clp_b', 'str_b', 'ten_b', 0, 1, 1, 1080, 1920, '/c', 'cut', ?)",
            (_T,),
        )
        await conn.execute(
            "INSERT INTO variants (id, clip_id, tenant_id, persona_id, language, caption, "
            "created_at) VALUES ('var_b', 'clp_b', 'ten_b', 'default', 'es', 'hola', ?)",
            (_T,),
        )
        await conn.execute(
            "INSERT INTO channel_watches (id, tenant_id, platform, channel_url, persona_id, "
            "created_at, updated_at) VALUES ('cw_c', 'ten_c', 'youtube', "
            "'https://youtube.com/@c', 'default', ?, ?)",
            (_T, _T),
        )
        await conn.commit()

        assert await apply_migrations(db) == 61
        # Idempotent.
        assert await apply_migrations(db) == 61

        cur = await conn.execute(
            "SELECT tenant_id, name, voice_prompt FROM personas WHERE id = 'default' "
            "ORDER BY tenant_id"
        )
        rows = [tuple(r) for r in await cur.fetchall()]
        assert rows[0] == ("ten_a", "A default", "A prompt")
        # B and C got a copy of the persona they were using; D the seed.
        assert rows[1] == ("ten_b", "A default", "A prompt")
        assert rows[2] == ("ten_c", "A default", "A prompt")
        assert rows[3][0:2] == ("ten_d", "Default")
        assert len(rows) == 4
        cur = await conn.execute("SELECT persona_id, tenant_id FROM variants")
        assert [tuple(r) for r in await cur.fetchall()] == [("default", "ten_b")]

        # The composite foreign key is enforced: a variant can only point at
        # its own account's persona.
        with pytest.raises(INTEGRITY_ERRORS):
            await conn.execute(
                "INSERT INTO variants (id, clip_id, tenant_id, persona_id, language, "
                "caption, created_at) VALUES ('var_x', 'clp_b', 'ten_b', 'nope', 'es', 'x', ?)",
                (_T,),
            )
        # Deleting an account's persona cascades only to its own variants.
        await conn.execute("DELETE FROM personas WHERE tenant_id = 'ten_a'")
        await conn.commit()
        cur = await conn.execute("SELECT count(*) FROM variants")
        assert (await cur.fetchone())[0] == 1
    finally:
        await db.close()
