"""Drive polling loop — voice-markers spec slice E.4.

Per-watch algorithm:

  1. Resolve the DriveClient (real Google impl or FakeDriveClient).
  2. `client.list_video_files(folder_id, modified_after=last_polled_at)`
  3. For each returned file whose id is NOT in `seen_file_ids`:
     - Download to `<output_dir>/.drive_inbox/<watch_id>/<file_id>/<name>`
     - Call `ingest_callback(tenant_id, local_path, original_name)`
       to enqueue the per-tenant ingest job.
     - Append file_id to `seen_file_ids`.
  4. Update `last_polled_at = now_iso()` and persist the updated
     `seen_file_ids` list.

Per-tenant idempotence — re-polling the same folder doesn't reingest
files we've already seen, because we filter by both `modified_after`
(server-side, cheap) AND `seen_file_ids` (client-side, exact).

Failures on a single file are logged and skipped; the rest of the
batch continues. A poll that completes partially still updates
`seen_file_ids` for the successful files so the next run picks up
where this one left off.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NamedTuple

import structlog

from chalybclip.db import Database, DriveWatchesRepo, TenantsRepo
from chalybclip.tenancy import bound_tenant

from .models import DriveWatchRow
from .protocol import DriveClient

_log = structlog.get_logger(__name__)


IngestCallback = Callable[[str, Path, str], Awaitable[None]]
"""(tenant_id, local_path, original_name) -> None.

The poller hands you the freshly-downloaded file; you decide what
ingest pipeline to enqueue. In production this wraps `ingest_vod`."""


DriveClientFactory = Callable[[DriveWatchRow], DriveClient]
"""Builds a DriveClient for one watch — given the row, hand back a
configured client. Lets us inject the OAuth state without dragging it
through every function signature."""


class PollReport(NamedTuple):
    """Per-watch outcome."""

    watch_id: str
    tenant_id: str
    folder_id: str
    files_seen: int
    files_ingested: int
    files_failed: int
    skipped_disabled: bool


# ---- public entry ----


async def poll_drive_watches(
    db: Database,
    *,
    output_dir: Path,
    drive_client_factory: DriveClientFactory,
    ingest_callback: IngestCallback,
    tenant_id: str | None = None,
) -> list[PollReport]:
    """Poll every enabled drive watch for new files.

    Args:
        db: Database handle. Migrations must already be applied.
        output_dir: Project output root. Downloads land under
            `<output_dir>/.drive_inbox/<watch_id>/<file_id>/<name>`.
        drive_client_factory: Builds a DriveClient per watch row.
        ingest_callback: Called for each newly-downloaded file with
            `(tenant_id, local_path, original_name)`.
        tenant_id: When set, restricts to that tenant; otherwise polls
            every tenant.

    Returns:
        One PollReport per watch row scanned.
    """
    output_dir = Path(output_dir).resolve()
    tenants_repo = TenantsRepo(db)

    if tenant_id is not None:
        t = await tenants_repo.get(tenant_id)
        tenants = [t] if t is not None else []
    else:
        tenants = await tenants_repo.list_all()

    reports: list[PollReport] = []
    for t in tenants:
        with bound_tenant(t.id):
            watches = await DriveWatchesRepo(db).list_for_tenant()
            for watch in watches:
                report = await _poll_one_watch(
                    db=db,
                    watch=watch,
                    output_dir=output_dir,
                    drive_client=drive_client_factory(watch),
                    ingest_callback=ingest_callback,
                )
                reports.append(report)
                _log.info(
                    "drive.poll_done",
                    watch_id=watch.id,
                    tenant_id=watch.tenant_id,
                    files_seen=report.files_seen,
                    files_ingested=report.files_ingested,
                    files_failed=report.files_failed,
                    skipped_disabled=report.skipped_disabled,
                )
    return reports


async def _poll_one_watch(
    *,
    db: Database,
    watch: DriveWatchRow,
    output_dir: Path,
    drive_client: DriveClient,
    ingest_callback: IngestCallback,
) -> PollReport:
    """Per-watch body. Tenant is already bound."""
    if not watch.enabled:
        return PollReport(
            watch_id=watch.id,
            tenant_id=watch.tenant_id,
            folder_id=watch.folder_id,
            files_seen=0,
            files_ingested=0,
            files_failed=0,
            skipped_disabled=True,
        )

    try:
        files = await drive_client.list_video_files(
            folder_id=watch.folder_id,
            modified_after=watch.last_polled_at,
        )
    except Exception as e:
        _log.warning(
            "drive.list_failed",
            watch_id=watch.id,
            folder_id=watch.folder_id,
            error=str(e),
        )
        return PollReport(
            watch_id=watch.id,
            tenant_id=watch.tenant_id,
            folder_id=watch.folder_id,
            files_seen=0,
            files_ingested=0,
            files_failed=1,
            skipped_disabled=False,
        )

    files_seen = len(files)
    files_ingested = 0
    files_failed = 0
    seen_ids = set(watch.seen_file_ids)
    inbox_root = output_dir / ".drive_inbox" / watch.id

    for f in files:
        if f.file_id in seen_ids:
            continue
        target = inbox_root / f.file_id / f.name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            await drive_client.download_file(
                file_id=f.file_id, target_path=target
            )
            await ingest_callback(watch.tenant_id, target, f.name)
            seen_ids.add(f.file_id)
            files_ingested += 1
        except Exception as e:
            _log.warning(
                "drive.file_failed",
                watch_id=watch.id,
                file_id=f.file_id,
                file_name=f.name,
                error=str(e),
            )
            files_failed += 1

    # Persist progress even on partial success. The seen_file_ids list
    # always gets the latest successes appended.
    #
    # `last_polled_at` only advances when every file in this batch
    # succeeded. Otherwise we leave it where it was so the next poll
    # re-lists the failed files (the server-side `modified_after`
    # filter won't hide them). `seen_file_ids` dedupes the successes
    # that already shipped, so re-listing them is cheap — we don't
    # download or call ingest_callback again.
    new_polled_at = _now_iso() if files_failed == 0 else watch.last_polled_at
    await DriveWatchesRepo(db).mark_polled(
        watch.id,
        seen_file_ids=sorted(seen_ids),
        last_polled_at=new_polled_at,
    )

    return PollReport(
        watch_id=watch.id,
        tenant_id=watch.tenant_id,
        folder_id=watch.folder_id,
        files_seen=files_seen,
        files_ingested=files_ingested,
        files_failed=files_failed,
        skipped_disabled=False,
    )


# ---- helpers ----


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()
