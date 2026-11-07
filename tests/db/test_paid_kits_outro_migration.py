"""Migration 062: existing paid brand kits stop appending the Chalyb end
card (owner, 2026-11-07). Free kits are left as they are, and running the
migration again changes nothing."""

from __future__ import annotations

import shutil
from pathlib import Path

from chalybclip.db import BrandKitsRepo, Database, TenantsRepo, apply_migrations
from chalybclip.tenancy import bound_tenant

from .._db_backend import migrated_database

_MIGRATIONS = Path(__file__).resolve().parents[2] / "chalybclip" / "db" / "migrations"
_SQL = (_MIGRATIONS / "062_paid_brand_kits_outro_off.sql").read_text(encoding="utf-8")

# Raw tier label → whether that tenant's kits must end up with the outro off.
_TIERS = {
    "free": False,
    "pro": True,
    "all_access": True,
    "VIP ": True,  # aliases + messy casing read as paid, like tiers.py
    "partner": True,
    "mystery": False,  # unknown labels read as free in tiers.py
}


async def _seed(db: Database) -> dict[str, str]:
    """One tenant per tier, each with a kit that still has the outro on."""
    kits: dict[str, str] = {}
    conn = await db.connect()
    for tier in _TIERS:
        tenant = (await TenantsRepo(db).create(name=f"t-{tier.strip()}")).id
        await conn.execute("UPDATE tenants SET tier = ? WHERE id = ?", (tier, tenant))
        with bound_tenant(tenant):
            kit = await BrandKitsRepo(db).create(
                name="K", primary_color="#fff", accent_color="#000",
                is_default=True, show_chalybclip_outro=True,
            )
        kits[tier] = kit.id
    await conn.commit()
    return kits


async def _outro(db: Database) -> dict[str, int]:
    conn = await db.connect()
    cur = await conn.execute("SELECT id, show_chalybclip_outro FROM brand_kits")
    return {r[0]: int(r[1]) for r in await cur.fetchall()}


async def test_paid_kits_flip_off_free_kits_untouched(tmp_path: Path) -> None:
    async for db in migrated_database(tmp_path):
        kits = await _seed(db)
        conn = await db.connect()
        await conn.executescript(_SQL)
        after = await _outro(db)
        for tier, paid in _TIERS.items():
            assert after[kits[tier]] == (0 if paid else 1), tier

        # Idempotent: a second run is a no-op.
        await conn.executescript(_SQL)
        assert await _outro(db) == after


async def test_paid_user_who_opts_back_in_is_not_reset(tmp_path: Path) -> None:
    async for db in migrated_database(tmp_path):
        kits = await _seed(db)
        conn = await db.connect()
        await conn.executescript(_SQL)
        await conn.execute(
            "UPDATE brand_kits SET show_chalybclip_outro = 1 WHERE id = ?", (kits["pro"],)
        )
        await conn.commit()
        # The runner never replays an applied migration.
        assert await apply_migrations(db) == 62
        assert (await _outro(db))[kits["pro"]] == 1


async def test_runner_applies_062_on_an_existing_v61_database(tmp_path: Path) -> None:
    """The deploy path: a database at v61 with live kits moves to v62."""
    old = tmp_path / "old_migrations"
    old.mkdir()
    for f in sorted(_MIGRATIONS.glob("*.sql")):
        if not f.name.endswith(".pg.sql") and int(f.name[:3]) <= 61:
            shutil.copy(f, old / f.name)
    db = Database(str(tmp_path / "old.db"))
    try:
        assert await apply_migrations(db, migrations_dir=old) == 61
        kits = await _seed(db)
        assert await apply_migrations(db) == 62
        after = await _outro(db)
        for tier, paid in _TIERS.items():
            assert after[kits[tier]] == (0 if paid else 1), tier
    finally:
        await db.close()


def test_tier_list_matches_tiers_module() -> None:
    """The SQL hand-copies the paid labels; keep it in step with tiers.py."""
    import re

    from chalybclip.tiers import _ALIASES, PAID_TIERS

    in_list = re.search(r"TRIM\(tier\)\) IN \(([^)]*)\)", _SQL)
    assert in_list is not None
    labels = set(re.findall(r"'([^']+)'", in_list.group(1)))
    expected = set(PAID_TIERS) | {a for a, t in _ALIASES.items() if t in PAID_TIERS}
    assert labels == expected
