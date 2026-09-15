"""Clip → Drive export service (task #31, mid-tier perk).

Pins the entitlement + upload contract with the FakeDriveUploadClient,
no network / OAuth:
  - pro + all_access (and partner, normalized) export successfully
  - free is NOT entitled → DriveExportNotEntitled (caller skips)
  - missing / empty source MP4 → DriveExportError
  - unknown clip → DriveExportError
  - the bytes actually land in the destination folder
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from chalybclip.db import (
    ClipsRepo,
    Database,
    StreamsRepo,
    TenantsRepo,
)
from chalybclip.db.models import ClipRow, StreamRow
from chalybclip.drive.export import (
    DriveExportError,
    DriveExportNotEntitled,
    export_clip_to_drive,
)
from chalybclip.drive.upload import FakeDriveUploadClient
from chalybclip.integrations.chalyb.service import sync_tenant_tier
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_tenant_with_clip(
    db: Database,
    *,
    tier: str | None,
    tmp_path: Path,
    clip_id: str = "clp_drv",
) -> tuple[str, Path]:
    """Create a tenant (optionally at `tier`) + a stream + a clip, and
    write a real MP4 on disk. Returns (tenant_id, source_mp4_path)."""
    tenant = await TenantsRepo(db).create(name="DriveCo")
    if tier is not None:
        await sync_tenant_tier(db, tenant_id=tenant.id, tier=tier)

    src = tmp_path / "clips" / clip_id / "clip_render_1080.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 50_000)

    with bound_tenant(tenant.id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id="str_drv",
                tenant_id=tenant.id,
                vod_url="x",
                platform="kick",
                title=None,
                channel=None,
                duration_s=30.0,
                source_video_path="/tmp/v",
                source_audio_path="/tmp/a",
                status="ingested",
                created_at=_now(),
            )
        )
        await ClipsRepo(db).upsert_many(
            [
                ClipRow(
                    id=clip_id,
                    stream_id="str_drv",
                    tenant_id=tenant.id,
                    candidate_id=None,
                    start_s=0.0,
                    end_s=30.0,
                    duration_s=30.0,
                    width=1080,
                    height=1920,
                    path=str(src.parent / "clip.mp4"),
                    status="cut",
                    created_at=_now(),
                )
            ]
        )
    return tenant.id, src


# ---- entitlement ----


@pytest.mark.parametrize("tier", ["pro", "all_access", "partner"])
async def test_paid_and_partner_tiers_export(
    drive_db: Database, tmp_path: Path, tier: str
) -> None:
    """pro (mid) + all_access (top) + partner (alias of all_access) all
    get Drive export."""
    tenant_id, src = await _seed_tenant_with_clip(
        drive_db, tier=tier, tmp_path=tmp_path,
    )
    client = FakeDriveUploadClient(dest_dir=tmp_path / "drive")

    outcome = await export_clip_to_drive(
        db=drive_db,
        tenant_id=tenant_id,
        clip_id="clp_drv",
        source_path=src,
        folder_id="folder_abc",
        client=client,
    )

    assert outcome.clip_id == "clp_drv"
    assert outcome.file_id.startswith("fakedrv_")
    # The bytes actually landed in the destination folder.
    assert client.uploads == [("folder_abc", "clp_drv.mp4", src.stat().st_size)]
    assert (tmp_path / "drive" / "folder_abc" / "clp_drv.mp4").is_file()


async def test_free_tier_not_entitled(
    drive_db: Database, tmp_path: Path
) -> None:
    """free tier → DriveExportNotEntitled. The post-render hook treats
    this as a silent skip, not a failure."""
    tenant_id, src = await _seed_tenant_with_clip(
        drive_db, tier=None, tmp_path=tmp_path,  # defaults to free
    )
    client = FakeDriveUploadClient(dest_dir=tmp_path / "drive")

    with pytest.raises(DriveExportNotEntitled):
        await export_clip_to_drive(
            db=drive_db,
            tenant_id=tenant_id,
            clip_id="clp_drv",
            source_path=src,
            folder_id="folder_abc",
            client=client,
        )
    # Nothing uploaded.
    assert client.uploads == []


# ---- error paths ----


async def test_missing_source_raises(
    drive_db: Database, tmp_path: Path
) -> None:
    tenant_id, _src = await _seed_tenant_with_clip(
        drive_db, tier="pro", tmp_path=tmp_path,
    )
    client = FakeDriveUploadClient(dest_dir=tmp_path / "drive")
    with pytest.raises(DriveExportError, match="source file missing"):
        await export_clip_to_drive(
            db=drive_db,
            tenant_id=tenant_id,
            clip_id="clp_drv",
            source_path=tmp_path / "does_not_exist.mp4",
            folder_id="folder_abc",
            client=client,
        )


async def test_empty_source_raises(
    drive_db: Database, tmp_path: Path
) -> None:
    tenant_id, src = await _seed_tenant_with_clip(
        drive_db, tier="pro", tmp_path=tmp_path,
    )
    src.write_bytes(b"")  # truncate to empty
    client = FakeDriveUploadClient(dest_dir=tmp_path / "drive")
    with pytest.raises(DriveExportError, match="empty"):
        await export_clip_to_drive(
            db=drive_db,
            tenant_id=tenant_id,
            clip_id="clp_drv",
            source_path=src,
            folder_id="folder_abc",
            client=client,
        )


async def test_unknown_clip_raises(
    drive_db: Database, tmp_path: Path
) -> None:
    tenant_id, src = await _seed_tenant_with_clip(
        drive_db, tier="pro", tmp_path=tmp_path,
    )
    client = FakeDriveUploadClient(dest_dir=tmp_path / "drive")
    with pytest.raises(DriveExportError, match="clip not found"):
        await export_clip_to_drive(
            db=drive_db,
            tenant_id=tenant_id,
            clip_id="clp_nonexistent",
            source_path=src,
            folder_id="folder_abc",
            client=client,
        )


async def test_custom_file_name_used(
    drive_db: Database, tmp_path: Path
) -> None:
    """Caller can override the destination filename (e.g. a human title
    instead of the clip id)."""
    tenant_id, src = await _seed_tenant_with_clip(
        drive_db, tier="all_access", tmp_path=tmp_path,
    )
    client = FakeDriveUploadClient(dest_dir=tmp_path / "drive")
    await export_clip_to_drive(
        db=drive_db,
        tenant_id=tenant_id,
        clip_id="clp_drv",
        source_path=src,
        folder_id="folder_abc",
        client=client,
        file_name="Mi Mejor Momento.mp4",
    )
    assert client.uploads[0][1] == "Mi Mejor Momento.mp4"
