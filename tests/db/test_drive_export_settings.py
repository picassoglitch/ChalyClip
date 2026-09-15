"""DriveExportSettingsRepo — per-tenant clip→Drive destination (task #31).

Pins: empty default, enabled toggle, destination + token writes,
is_connected gating, disconnect wipe, and tenant isolation.
"""

from __future__ import annotations

import pytest

from chalybclip.db import Database, DriveExportSettingsRepo, TenantsRepo
from chalybclip.tenancy import bound_tenant


async def test_get_returns_none_before_any_touch(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    with bound_tenant(tenant.id):
        assert await DriveExportSettingsRepo(migrated_db).get() is None


async def test_set_enabled_creates_row_and_toggles(
    migrated_db: Database,
) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    with bound_tenant(tenant.id):
        repo = DriveExportSettingsRepo(migrated_db)
        row = await repo.set_enabled(True)
        assert row.enabled is True
        assert row.tenant_id == tenant.id
        # Not connected yet — enabled alone doesn't mean exports can run.
        assert row.is_connected is False
        row = await repo.set_enabled(False)
        assert row.enabled is False


async def test_destination_and_tokens_make_it_connected(
    migrated_db: Database,
) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    with bound_tenant(tenant.id):
        repo = DriveExportSettingsRepo(migrated_db)
        await repo.set_destination(folder_id="fld_1", folder_name="Clips")
        # Folder but no token → still not connected.
        row = await repo.get()
        assert row is not None and row.is_connected is False
        # Add the OAuth refresh token → connected.
        row = await repo.set_tokens(
            refresh_token="rt_abc",
            access_token="at_xyz",
            access_token_expires_at="2026-01-01T00:00:00+00:00",
        )
        assert row.is_connected is True
        assert row.folder_id == "fld_1"
        assert row.refresh_token == "rt_abc"


async def test_disconnect_wipes_credentials_and_disables(
    migrated_db: Database,
) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    with bound_tenant(tenant.id):
        repo = DriveExportSettingsRepo(migrated_db)
        await repo.set_destination(folder_id="fld_1", folder_name="Clips")
        await repo.set_tokens(refresh_token="rt_abc")
        await repo.set_enabled(True)
        assert (await repo.get()).is_connected is True

        await repo.disconnect()
        row = await repo.get()
        assert row is not None
        assert row.is_connected is False
        assert row.refresh_token is None
        assert row.folder_id is None
        assert row.enabled is False


async def test_isolated_per_tenant(migrated_db: Database) -> None:
    alice = await TenantsRepo(migrated_db).create(name="Alice")
    bob = await TenantsRepo(migrated_db).create(name="Bob")
    with bound_tenant(alice.id):
        await DriveExportSettingsRepo(migrated_db).set_enabled(True)
    with bound_tenant(bob.id):
        # Bob never touched it → None, unaffected by Alice.
        assert await DriveExportSettingsRepo(migrated_db).get() is None
