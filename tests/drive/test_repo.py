"""DriveWatchesRepo — CRUD + tenant isolation (slice E.4)."""

from __future__ import annotations

from chalybclip.db import Database, DriveWatchesRepo, TenantsRepo
from chalybclip.errors import ChalybClipError
from chalybclip.tenancy import bound_tenant


async def test_create_and_get(drive_db: Database) -> None:
    t = await TenantsRepo(drive_db).create(name="A")
    with bound_tenant(t.id):
        watch = await DriveWatchesRepo(drive_db).create(
            folder_id="folder_xyz",
            folder_name="ChalybClip Inbox",
            refresh_token="rt-1",
        )
        assert watch.id.startswith("drv_")
        assert watch.folder_id == "folder_xyz"
        assert watch.folder_name == "ChalybClip Inbox"
        assert watch.refresh_token == "rt-1"
        assert watch.seen_file_ids == []
        assert watch.enabled is True

        looked_up = await DriveWatchesRepo(drive_db).get(watch.id)
    assert looked_up is not None
    assert looked_up.folder_id == "folder_xyz"


async def test_list_for_tenant_orders_by_created_at(drive_db: Database) -> None:
    t = await TenantsRepo(drive_db).create(name="A")
    with bound_tenant(t.id):
        repo = DriveWatchesRepo(drive_db)
        a = await repo.create(folder_id="f1", folder_name=None, refresh_token="r1")
        b = await repo.create(folder_id="f2", folder_name=None, refresh_token="r2")
        watches = await repo.list_for_tenant()
    assert [w.id for w in watches] == [a.id, b.id]


async def test_mark_polled_persists_seen_file_ids(drive_db: Database) -> None:
    t = await TenantsRepo(drive_db).create(name="A")
    with bound_tenant(t.id):
        repo = DriveWatchesRepo(drive_db)
        watch = await repo.create(
            folder_id="f", folder_name=None, refresh_token="r"
        )
        await repo.mark_polled(
            watch.id,
            seen_file_ids=["file_a", "file_b"],
            last_polled_at="2026-05-13T12:00:00+00:00",
        )
        updated = await repo.get(watch.id)
    assert updated is not None
    assert updated.seen_file_ids == ["file_a", "file_b"]
    assert updated.last_polled_at == "2026-05-13T12:00:00+00:00"


async def test_set_enabled_toggles_without_losing_history(drive_db: Database) -> None:
    """Pausing a watch keeps seen_file_ids so resuming doesn't reingest."""
    t = await TenantsRepo(drive_db).create(name="A")
    with bound_tenant(t.id):
        repo = DriveWatchesRepo(drive_db)
        watch = await repo.create(
            folder_id="f", folder_name=None, refresh_token="r"
        )
        await repo.mark_polled(
            watch.id,
            seen_file_ids=["seen-1"],
            last_polled_at="2026-05-13T12:00:00+00:00",
        )
        paused = await repo.set_enabled(watch.id, False)
        assert paused.enabled is False
        assert paused.seen_file_ids == ["seen-1"]
        resumed = await repo.set_enabled(watch.id, True)
    assert resumed.enabled is True
    assert resumed.seen_file_ids == ["seen-1"]


async def test_set_enabled_unknown_raises(drive_db: Database) -> None:
    t = await TenantsRepo(drive_db).create(name="A")
    with bound_tenant(t.id):
        repo = DriveWatchesRepo(drive_db)
        import pytest

        with pytest.raises(ChalybClipError, match="not found"):
            await repo.set_enabled("drv_nope", True)


async def test_delete_removes_row(drive_db: Database) -> None:
    t = await TenantsRepo(drive_db).create(name="A")
    with bound_tenant(t.id):
        repo = DriveWatchesRepo(drive_db)
        watch = await repo.create(
            folder_id="f", folder_name=None, refresh_token="r"
        )
        await repo.delete(watch.id)
        assert await repo.get(watch.id) is None


async def test_isolated_per_tenant(drive_db: Database) -> None:
    alice = await TenantsRepo(drive_db).create(name="Alice")
    bob = await TenantsRepo(drive_db).create(name="Bob")
    with bound_tenant(alice.id):
        a_watch = await DriveWatchesRepo(drive_db).create(
            folder_id="alice_folder", folder_name=None, refresh_token="r-a"
        )
    with bound_tenant(bob.id):
        assert await DriveWatchesRepo(drive_db).get(a_watch.id) is None
        assert await DriveWatchesRepo(drive_db).list_for_tenant() == []
