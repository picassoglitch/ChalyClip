"""Export a rendered clip to a tenant's Google Drive — mid-tier perk.

Task #31. `pro` (mid tier) and `all_access` (top tier) can opt to have
each rendered clip auto-saved to their connected Drive folder; `free`
downloads locally only.

This module is the SERVICE layer — pure, testable, no FastAPI, no real
Google calls. It:

  1. Checks the tenant is entitled (tier in PAID_TIERS — pro or
     all_access). The tier is read from the DB and normalized via
     chalybclip.tiers, so this works from the background post-render hook
     where there's no request.state to read.
  2. Validates the source MP4 exists + is non-empty.
  3. Delegates the actual upload to an injected `DriveUploadClient`
     (Fake in tests / dev, real GoogleDriveUploadClient in prod — the
     latter deferred until the Google Cloud OAuth app is set up).
  4. Emits a structured event for the dashboard + returns an outcome.

The CLIENT (which holds the tenant's OAuth token) and the DESTINATION
folder_id are passed IN — resolving those from per-tenant Drive
credentials is the connect-flow follow-up. Keeping them as parameters
means this core is fully exercised by tests today and the persistence
layer slots in without touching this logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chalybclip.db import Database
from chalybclip.errors import ChalybClipError
from chalybclip.tenancy import bound_tenant
from chalybclip.tiers import PAID_TIERS, normalize_tier

from .upload import DriveUploadClient, DriveUploadResult


class DriveExportError(ChalybClipError):
    """Base for clip→Drive export failures."""


class DriveExportNotEntitled(DriveExportError):
    """Tenant's tier doesn't include Drive export (free tier)."""


@dataclass(frozen=True, slots=True)
class DriveExportOutcome:
    clip_id: str
    file_id: str
    web_view_link: str | None
    size_bytes: int | None


async def export_clip_to_drive(
    *,
    db: Database,
    tenant_id: str,
    clip_id: str,
    source_path: Path,
    folder_id: str,
    client: DriveUploadClient,
    file_name: str | None = None,
) -> DriveExportOutcome:
    """Upload a rendered clip MP4 to the tenant's Drive folder.

    Raises:
      DriveExportNotEntitled — tenant tier is below pro (Drive export is
        a paid perk). Caller (the post-render hook) treats this as a
        no-op skip, NOT an error.
      DriveExportError — clip missing / not owned by tenant, source MP4
        missing or empty, or the upload itself failed.

    On success emits `drive.export.succeeded`; on a failed UPLOAD emits
    `drive.export.failed` before re-raising so the dashboard can show
    it. Entitlement skips are silent (no event) — they're expected for
    free tenants, not failures.
    """
    from chalybclip.db import ClipsRepo, EventsRepo

    with bound_tenant(tenant_id):
        # ---- entitlement (tier) ----
        from chalybclip.db import TenantsRepo

        tenant = await TenantsRepo(db).get(tenant_id)
        tier = normalize_tier(tenant.tier if tenant else None)
        if tier not in PAID_TIERS:
            raise DriveExportNotEntitled(
                f"tenant tier '{tier}' does not include Drive export "
                "(requires pro or all_access)"
            )

        # ---- clip ownership ----
        clip = await ClipsRepo(db).get(clip_id)
        if clip is None:
            raise DriveExportError(f"clip not found: {clip_id}")

        # ---- source file ----
        src = Path(source_path)
        if not src.is_file():
            raise DriveExportError(
                f"source file missing for clip {clip_id}: {src}"
            )
        if src.stat().st_size == 0:
            raise DriveExportError(
                f"source file is empty for clip {clip_id}: {src}"
            )

        name = file_name or f"{clip_id}.mp4"

        # ---- upload ----
        try:
            result: DriveUploadResult = await client.upload_file(
                local_path=src,
                folder_id=folder_id,
                name=name,
            )
        except DriveExportError:
            raise
        except Exception as e:  # noqa: BLE001 — wrap any client error
            try:
                await EventsRepo(db).emit(
                    type="drive.export.failed",
                    payload={
                        "clip_id": clip_id,
                        "folder_id": folder_id,
                        "error": str(e)[:500],
                    },
                )
            except Exception:  # noqa: BLE001 — event emit never blocks
                pass
            raise DriveExportError(
                f"Drive upload failed for clip {clip_id}: {e}"
            ) from e

        await EventsRepo(db).emit(
            type="drive.export.succeeded",
            payload={
                "clip_id": clip_id,
                "folder_id": folder_id,
                "file_id": result.file_id,
                "size_bytes": result.size_bytes,
            },
        )

    return DriveExportOutcome(
        clip_id=clip_id,
        file_id=result.file_id,
        web_view_link=result.web_view_link,
        size_bytes=result.size_bytes,
    )


__all__ = [
    "DriveExportError",
    "DriveExportNotEntitled",
    "DriveExportOutcome",
    "export_clip_to_drive",
]
