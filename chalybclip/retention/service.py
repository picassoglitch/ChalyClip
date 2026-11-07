"""Retention sweeper — hard-deletes a tenant's data past its plan window.

Window per plan (`chalybclip.tiers.RETENTION_DAYS`, mirroring the hub's
TIER_CAPS): free 7 days, pro / all_access (Partner, VIP) 30 days, for
clips, VODs and transcripts alike. A tenant's explicit `retention_*_days`
column overrides its plan; an unrecognized tier gets the longest window.

Per tenant, in order:
  1. Clips older than the window: bucket objects under
     `clips/<tenant>/<clip>/`, then the local clip dir, then the row.
     Skipped while rendering, with a publish pending, or while their
     stream has a pipeline run in flight.
  2. Transcripts older than the window (rows only).
  3. Streams older than the window: the source (bucket `uploads/` +
     `work/` prefixes and local files) goes; the row and per-stream dir go
     once no clip or transcript survives under them.

Timestamps are TEXT, so they're parsed and compared as datetimes. A bucket
delete failure keeps the row so the next run retries. `dry_run=True`
counts without touching rows, files or the bucket.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import structlog

from chalybclip.db import Database, TenantsRepo
from chalybclip.tenancy import bound_tenant
from chalybclip.tiers import (
    LONGEST_RETENTION_DAYS,
    resolve_tier_alias,
    retention_days,
)

if TYPE_CHECKING:
    from chalybclip.integrations.storage import ArtifactStore

_log = structlog.get_logger(__name__)

# Pending publish states (same set the stream-deletion guard uses): a
# scheduled post fetches our media when its slot fires, so its clip must stay.
_PENDING_PUBLISH_STATES = ("scheduled", "pending", "queued", "publishing", "processing")
# publish_jobs rows that are done; anything else may still need the clip.
_DONE_PUBLISH_JOB_STATES = ("sent", "failed", "canceled", "cancelled", "published")

DEFAULT_BATCH_SIZE = 200
# A stream whose usage job moved within this many hours may still be running.
_BUSY_GRACE_HOURS = 24


class RetentionPolicy(NamedTuple):
    """Resolved per-tenant windows (days).

    Each window is the tenant's explicit column override when set, else the
    plan window for its tier (`chalybclip.tiers.RETENTION_DAYS`). A tier we
    don't recognize gets the LONGEST plan window — never free's short one.
    """

    vod_days: int
    clip_days: int
    transcript_days: int
    tier: str = ""
    tier_known: bool = True

    @classmethod
    def from_tenant(
        cls,
        *,
        tier: str | None,
        vod_days: int | None = None,
        clip_days: int | None = None,
        transcript_days: int | None = None,
    ) -> RetentionPolicy:
        plan_days = retention_days(tier)
        known = plan_days is not None
        window = plan_days if plan_days is not None else LONGEST_RETENTION_DAYS

        # A stored 0 or negative would put the cutoff at (or after) now and
        # wipe the tenant; treat it as unset.
        def pick(override: int | None) -> int:
            return override if override is not None and override > 0 else window

        return cls(
            vod_days=pick(vod_days),
            clip_days=pick(clip_days),
            transcript_days=pick(transcript_days),
            tier=resolve_tier_alias(tier) or (tier or ""),
            tier_known=known,
        )


class RetentionReport(NamedTuple):
    """Per-tenant outcome — what was deleted (or would be, if dry-run)."""

    tenant_id: str
    policy: RetentionPolicy
    vods_deleted: int
    clips_deleted: int
    transcripts_deleted: int
    bytes_freed: int
    dry_run: bool
    streams_deleted: int = 0
    objects_deleted: int = 0
    skipped_in_flight: int = 0
    skipped_bad_timestamp: int = 0
    failed: int = 0


# ---- helpers ----


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _cutoff(days_back: int) -> _dt.datetime:
    return _utc_now() - _dt.timedelta(days=days_back)


def _parse_ts(raw: object) -> _dt.datetime | None:
    """Parse a TEXT `created_at` into an aware UTC datetime, or None.

    Compared as datetimes, not strings: `Z` vs `+00:00`, a space separator
    or a naive value sort wrong lexically. Naive values are taken as UTC.
    """
    if isinstance(raw, _dt.datetime):
        dt = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        if text[-1] in "Zz":
            text = text[:-1] + "+00:00"
        try:
            dt = _dt.datetime.fromisoformat(text)
        except ValueError:
            return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=_dt.UTC)


def _safe_unlink(path: Path) -> int:
    """Delete `path` if it exists. Returns bytes freed (0 when missing).

    Idempotent — a re-run after a partial sweep doesn't trip on already-
    removed files. We track bytes freed for the report so operators can
    see how much disk the sweep recovered."""
    try:
        if path.is_file():
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            return size
        # Directory: walk + sum. Used for `<stream_dir>` and `<clip_dir>`.
        if path.is_dir():
            total = 0
            for child in path.rglob("*"):
                if child.is_file():
                    with contextlib.suppress(OSError):
                        total += child.stat().st_size
            # rmtree-ish, but tolerant of missing children racing concurrent
            # processes (e.g. operator manually deleting clips mid-sweep).
            for child in sorted(path.rglob("*"), reverse=True):
                with contextlib.suppress(OSError):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
            with contextlib.suppress(OSError):
                path.rmdir()
            return total
    except OSError as e:
        _log.warning("retention.unlink_failed", path=str(path), error=str(e))
    return 0


# ---- public entry ----


def reclaim_stream_source(
    *,
    stream_dir: Path,
    source_video_path: Path | None = None,
    source_audio_path: Path | None = None,
) -> int:
    """Delete a stream's raw source artifacts (downloaded video + extracted
    audio), returning bytes freed.

    Idempotent — safe to call when the source is already gone (returns 0).
    The conventional layout keeps both under ``<stream_dir>/source/``; live
    recordings (and legacy layouts) put the source files outside the
    per-stream dir, so we unlink those explicit paths too when they live
    elsewhere. Used by the pipeline's delete-on-completion path; the
    retention sweeper reaches the same files via the per-stream dir.
    """
    src_dir = stream_dir / "source"
    freed = _safe_unlink(src_dir)
    for p in (source_video_path, source_audio_path):
        if p is not None and src_dir not in p.parents:
            freed += _safe_unlink(p)
    return freed


def _source_recently_active(
    *,
    stream_dir: Path,
    video: Path | None,
    audio: Path | None,
    now: float,
    grace_s: float,
) -> bool:
    """True if a stream's source is being ACTIVELY written — an in-flight
    download we must never delete.

    Robust to HLS/DASH (Twitch/Kick/YouTube VODs): those have NO final
    `video.mp4` yet while downloading, only `video.mp4.part-FragNNNN`
    fragments, so checking `video.mp4`'s mtime alone misses them and the
    reclaimer would yank the dir out mid-download (yt-dlp then floods
    'Unable to rename ...part-FragNNNN'). We instead look at the whole
    `source/` dir: its own mtime (churns as fragments are created/renamed)
    and the newest mtime among its files. Short-circuits on the first
    recent thing it finds.
    """
    src_dir = stream_dir / "source"
    for p in (src_dir, video, audio):
        if p is None:
            continue
        try:
            if now - p.stat().st_mtime < grace_s:
                return True
        except OSError:
            continue
    try:
        with os.scandir(src_dir) as it:
            for entry in it:
                try:
                    if now - entry.stat().st_mtime < grace_s:
                        return True
                except OSError:
                    continue
    except OSError:
        pass
    return False


async def reclaim_sources_until_free(
    db: Database,
    *,
    output_dir: Path,
    target_free_bytes: int,
    active_grace_s: float = 600.0,
) -> tuple[int, bool]:
    """Disk-budget enforcer: evict the OLDEST raw stream sources (video +
    audio) until the output volume has at least `target_free_bytes` free.

    This is what keeps the volume from filling regardless of user/stream
    count — time-window retention only bounds AGE, not total size. It is
    deliberately conservative:
      * Sources ONLY. Never touches clips/transcripts (user output) — those
        stay on their own retention windows. If shedding every source still
        can't reach the target, we log loudly: the operator needs a bigger
        volume or object-storage offload, not more deletion.
      * Oldest-first, so the most recently-active streams are last to go.
      * Skips any source whose `source/` dir saw activity within
        `active_grace_s` — an in-flight download (incl. an HLS/DASH fetch that
        has only `.part-Frag*` fragments and no `video.mp4` yet) we must not
        yank out from under. See `_source_recently_active`.

    Returns (bytes_freed, target_reached). No-op (0, True) when already above
    target or when target_free_bytes <= 0.
    """
    output_dir = Path(output_dir).resolve()
    if target_free_bytes <= 0:
        return 0, True
    try:
        free = shutil.disk_usage(output_dir).free
    except OSError:
        return 0, True
    if free >= target_free_bytes:
        return 0, True

    from chalybclip.jobs.active import active_stream_ids

    conn = await db.connect()
    cur = await conn.execute(
        "SELECT id, source_video_path, source_audio_path FROM streams "
        "ORDER BY created_at ASC"
    )
    rows = await cur.fetchall()
    now = time.time()
    freed = 0
    in_flight = active_stream_ids()
    for row in rows:
        try:
            if shutil.disk_usage(output_dir).free >= target_free_bytes:
                break
        except OSError:
            break
        # Never yank a source out from under a RUNNING pipeline. The mtime
        # check below can't see it: a multi-hour transcribe/LLM stretch only
        # READS source/, so the dir goes "idle" minutes into the run.
        if row["id"] in in_flight:
            continue
        video = Path(row["source_video_path"]) if row["source_video_path"] else None
        audio = Path(row["source_audio_path"]) if row["source_audio_path"] else None
        stream_dir = output_dir / row["id"]
        # Don't reclaim a source that's still being written (active download) —
        # incl. an HLS/DASH fetch that has only .part-Frag* files, no video.mp4.
        if _source_recently_active(
            stream_dir=stream_dir, video=video, audio=audio,
            now=now, grace_s=active_grace_s,
        ):
            continue
        freed += reclaim_stream_source(
            stream_dir=stream_dir,
            source_video_path=video,
            source_audio_path=audio,
        )

    try:
        reached = shutil.disk_usage(output_dir).free >= target_free_bytes
    except OSError:
        reached = False
    if freed:
        _log.info(
            "retention.reclaim_until_free",
            bytes_freed=freed, target_bytes=target_free_bytes, reached=reached,
        )
    if not reached:
        _log.warning(
            "retention.reclaim_until_free.target_unmet",
            target_bytes=target_free_bytes,
            hint="all reclaimable sources shed and still low — grow the "
                 "volume or enable object-storage offload",
        )
    return freed, reached


async def reclaim_orphan_dirs(
    db: Database,
    *,
    output_dir: Path,
    grace_s: float = 24 * 3600.0,
) -> tuple[int, int]:
    """Delete per-stream dirs on disk that have NO `streams` row left.

    Every other reclaimer here is DB-driven — it walks stream ROWS and
    unlinks the files they point to. A directory whose row is gone (the
    SQLite -> Postgres cutover stranded 188 dirs / ~266 GB of these; a
    row hard-deleted while its unlink partially failed leaves more) is
    invisible to all of them forever, so the volume ratchets up with
    garbage no window can age out. This pass closes that gap: anything
    under `output_dir` shaped like a stream dir that no row claims gets
    reclaimed.

    Safety rails, in order:
      * Refuses to act when the streams table is EMPTY — a populated
        volume next to an empty table smells like a mispointed
        DATABASE_URL, not 100% garbage. Deleting everything on that
        signal would be catastrophic; log and bail instead.
      * Skips ids registered in `jobs.active` (a run in flight whose row
        was deleted mid-run must keep its working dir).
      * Skips dirs with ANY write inside `grace_s` — an in-flight
        download can exist moments before its row lands, and mtime is
        the only signal that covers every such race.
      * Only `str_*` dirs directly under `output_dir`; never files, never
        other layouts.

    Returns (dirs_deleted, bytes_freed).
    """
    from chalybclip.jobs.active import active_stream_ids

    output_dir = Path(output_dir).resolve()
    conn = await db.connect()
    cur = await conn.execute("SELECT id FROM streams")
    db_ids = {row["id"] for row in await cur.fetchall()}
    if not db_ids:
        _log.warning(
            "retention.orphan_sweep_refused",
            reason="streams table is empty — refusing to treat the whole "
                   "volume as orphaned (mispointed DATABASE_URL?)",
        )
        return 0, 0

    in_flight = active_stream_ids()
    now = time.time()
    deleted = 0
    freed = 0
    try:
        entries = sorted(os.listdir(output_dir))
    except OSError:
        return 0, 0
    for name in entries:
        if not name.startswith("str_") or name in db_ids or name in in_flight:
            continue
        path = output_dir / name
        if not path.is_dir():
            continue
        newest = 0.0
        with contextlib.suppress(OSError):
            newest = path.stat().st_mtime
        for child in path.rglob("*"):
            with contextlib.suppress(OSError):
                newest = max(newest, child.stat().st_mtime)
        if now - newest < grace_s:
            continue
        freed += _safe_unlink(path)
        deleted += 1
    if deleted:
        _log.info(
            "retention.orphan_sweep_done",
            dirs_deleted=deleted, bytes_freed=freed,
        )
    return deleted, freed




async def sweep_retention(
    db: Database,
    *,
    output_dir: Path,
    tenant_id: str | None = None,
    dry_run: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[RetentionReport]:
    """Sweep per-plan retention windows for every tenant (or one).

    Args:
        db: Database handle. Migrations must already be applied.
        output_dir: Project output root. Per-stream dirs live at
            `<output_dir>/<stream_id>/`.
        tenant_id: When set, restricts the sweep to that tenant.
        dry_run: Count what would be deleted; touch no row, file or object.
        batch_size: Rows deleted between commits / progress logs.

    Returns:
        One RetentionReport per tenant scanned.
    """
    output_dir = Path(output_dir).resolve()
    tenants_repo = TenantsRepo(db)
    if tenant_id is not None:
        t = await tenants_repo.get(tenant_id)
        tenants = [t] if t is not None else []
    else:
        tenants = await tenants_repo.list_all()

    artifact_store: ArtifactStore | None = None
    store_broken = False
    if tenants and not dry_run:
        try:
            from chalybclip.integrations.storage import build_artifact_store
            from chalybclip.settings import get_settings

            artifact_store = build_artifact_store(get_settings())
        except Exception as e:  # process boundary: report, keep the rows
            _log.error("retention.artifact_store_unavailable", error=str(e))
            store_broken = True

    reports: list[RetentionReport] = []
    by_tier: dict[str, dict[str, int]] = {}
    for t in tenants:
        policy = RetentionPolicy.from_tenant(
            tier=t.tier,
            vod_days=t.retention_vod_days,
            clip_days=t.retention_clip_days,
            transcript_days=t.retention_transcript_days,
        )
        if not policy.tier_known:
            _log.warning(
                "retention.unknown_tier",
                tenant_id=t.id,
                tier=t.tier,
                window_days=LONGEST_RETENTION_DAYS,
            )
        try:
            with bound_tenant(t.id):
                report = await _sweep_one_tenant(
                    db=db,
                    tenant_id=t.id,
                    output_dir=output_dir,
                    policy=policy,
                    dry_run=dry_run,
                    artifact_store=artifact_store,
                    store_broken=store_broken,
                    batch_size=max(1, batch_size),
                )
        except Exception as e:  # one bad tenant must not stall the others
            _log.error("retention.tenant_failed", tenant_id=t.id, error=str(e))
            with contextlib.suppress(Exception):
                await (await db.connect()).rollback()
            continue
        reports.append(report)
        _log.info(
            "retention.sweep_done",
            tenant_id=t.id,
            tier=policy.tier,
            tier_known=policy.tier_known,
            clip_days=policy.clip_days,
            vod_days=policy.vod_days,
            transcript_days=policy.transcript_days,
            vods_deleted=report.vods_deleted,
            streams_deleted=report.streams_deleted,
            clips_deleted=report.clips_deleted,
            transcripts_deleted=report.transcripts_deleted,
            objects_deleted=report.objects_deleted,
            skipped_in_flight=report.skipped_in_flight,
            skipped_bad_timestamp=report.skipped_bad_timestamp,
            failed=report.failed,
            bytes_freed=report.bytes_freed,
            dry_run=dry_run,
        )
        agg = by_tier.setdefault(policy.tier or "unknown", {})
        for field in (
            "clips_deleted", "streams_deleted", "transcripts_deleted",
            "objects_deleted", "skipped_in_flight", "failed",
        ):
            agg[field] = agg.get(field, 0) + int(getattr(report, field))
        agg["tenants"] = agg.get("tenants", 0) + 1
    _log.info("retention.sweep_summary", by_tier=by_tier, dry_run=dry_run)
    return reports


def _owned_clip_dir(raw: object, clip_id: str, output_dir: Path) -> Path | None:
    """The clip's own dir (`.../<clip_id>/clip.mp4`) under `output_dir`, or
    None: an odd legacy path must never rmtree a shared or parent dir."""
    if not raw:
        return None
    parent = Path(str(raw)).parent
    if parent.name != clip_id:
        return None
    try:
        resolved = parent.resolve()
    except OSError:
        return None
    if output_dir not in resolved.parents:
        return None
    return resolved


async def _delete_prefixes(store: ArtifactStore, prefixes: list[str]) -> int:
    total = 0
    for prefix in prefixes:
        total += await store.delete_prefix(prefix=prefix)
    return total


async def _busy_stream_ids(conn: Any, tenant_id: str) -> set[str]:
    """Streams with a run in flight, read from the DB: the in-process registry
    is empty in the scheduled job's own process. A usage job touched within
    the grace window or a pending/live stream counts as busy."""
    from chalybclip.jobs.active import active_stream_ids

    busy = set(active_stream_ids())
    cur = await conn.execute(
        "SELECT id FROM streams WHERE tenant_id = ? AND status IN ('pending', 'live')",
        (tenant_id,),
    )
    busy.update(str(r["id"]) for r in await cur.fetchall())
    recent = _utc_now() - _dt.timedelta(hours=_BUSY_GRACE_HOURS)
    cur = await conn.execute(
        "SELECT stream_id, updated_at FROM usage_jobs WHERE tenant_id = ?",
        (tenant_id,),
    )
    for r in await cur.fetchall():
        touched = _parse_ts(r["updated_at"])
        if touched is None or touched >= recent:
            busy.add(str(r["stream_id"]))
    return busy


async def _in_flight_clip_ids(conn: Any, tenant_id: str) -> set[str]:
    """Clips with a publish still pending — deleting them breaks the post."""
    ids: set[str] = set()
    placeholders = ", ".join("?" for _ in _PENDING_PUBLISH_STATES)
    cur = await conn.execute(
        "SELECT clip_id FROM zernio_publishes WHERE tenant_id = ? "
        f"AND COALESCE(status, '') IN ('', {placeholders})",
        (tenant_id, *_PENDING_PUBLISH_STATES),
    )
    ids.update(str(r["clip_id"]) for r in await cur.fetchall())
    done = ", ".join("?" for _ in _DONE_PUBLISH_JOB_STATES)
    cur = await conn.execute(
        "SELECT clip_id FROM publish_jobs WHERE tenant_id = ? "
        f"AND COALESCE(status, '') NOT IN ({done})",
        (tenant_id, *_DONE_PUBLISH_JOB_STATES),
    )
    ids.update(str(r["clip_id"]) for r in await cur.fetchall())
    return ids


async def _sweep_one_tenant(
    *,
    db: Database,
    tenant_id: str,
    output_dir: Path,
    policy: RetentionPolicy,
    dry_run: bool,
    artifact_store: ArtifactStore | None,
    store_broken: bool,
    batch_size: int,
) -> RetentionReport:
    """The per-tenant body of `sweep_retention`. Tenant is already bound.

    Per expired clip: bucket objects first, then local files, then the row —
    if the bucket delete fails the row stays, so the next run retries
    instead of leaving objects nothing points at.
    """
    from chalybclip.integrations.storage import clip_prefix, stream_prefixes

    vod_cutoff = _cutoff(policy.vod_days)
    clip_cutoff = _cutoff(policy.clip_days)
    transcript_cutoff = _cutoff(policy.transcript_days)
    conn = await db.connect()
    in_flight_streams = await _busy_stream_ids(conn, tenant_id)

    bytes_freed = 0
    vods_deleted = 0
    streams_deleted = 0
    clips_deleted = 0
    transcripts_deleted = 0
    objects_deleted = 0
    skipped_in_flight = 0
    skipped_bad_ts = 0
    failed = 0
    since_commit = 0

    async def _tick() -> None:
        nonlocal since_commit
        since_commit += 1
        if since_commit >= batch_size and not dry_run:
            await conn.commit()
            _log.info("retention.batch_committed", tenant_id=tenant_id, rows=since_commit)
            since_commit = 0

    # --- Clips.
    cur = await conn.execute(
        "SELECT id, stream_id, path, render_state, created_at FROM clips "
        "WHERE tenant_id = ?",
        (tenant_id,),
    )
    clip_rows = list(await cur.fetchall())
    pending_publish = await _in_flight_clip_ids(conn, tenant_id)
    gone_clips: set[str] = set()  # deleted (or would be, in a dry run)
    clips_by_stream: dict[str, list[str]] = {}
    for row in clip_rows:
        clip_id = str(row["id"])
        clips_by_stream.setdefault(str(row["stream_id"]), []).append(clip_id)
        created = _parse_ts(row["created_at"])
        if created is None:
            skipped_bad_ts += 1
            _log.warning("retention.bad_timestamp", table="clips", id=clip_id)
            continue
        if created >= clip_cutoff:
            continue
        if (
            row["render_state"] == "rendering"
            or clip_id in pending_publish
            or row["stream_id"] in in_flight_streams
        ):
            skipped_in_flight += 1
            continue
        if dry_run:
            gone_clips.add(clip_id)
            clips_deleted += 1
            continue
        if store_broken:
            failed += 1
            continue
        if artifact_store is not None:
            try:
                objects_deleted += await _delete_prefixes(
                    artifact_store, [clip_prefix(tenant_id, clip_id)]
                )
            except Exception as e:
                failed += 1
                _log.warning(
                    "retention.clip_objects_failed",
                    tenant_id=tenant_id, clip_id=clip_id, error=str(e),
                )
                continue
        clip_dir = _owned_clip_dir(row["path"], clip_id, output_dir)
        if clip_dir is not None:
            bytes_freed += _safe_unlink(clip_dir)
        await conn.execute(
            "DELETE FROM clips WHERE id = ? AND tenant_id = ?",
            (clip_id, tenant_id),
        )
        gone_clips.add(clip_id)
        clips_deleted += 1
        await _tick()

    # --- Transcripts (row-only — they live in the DB).
    cur = await conn.execute(
        "SELECT stream_id, created_at FROM transcripts WHERE tenant_id = ?",
        (tenant_id,),
    )
    transcript_streams: set[str] = set()
    gone_transcripts: set[str] = set()
    for row in await cur.fetchall():
        sid = str(row["stream_id"])
        transcript_streams.add(sid)
        created = _parse_ts(row["created_at"])
        if created is None:
            skipped_bad_ts += 1
            _log.warning("retention.bad_timestamp", table="transcripts", id=sid)
            continue
        if created >= transcript_cutoff:
            continue
        if sid in in_flight_streams:
            skipped_in_flight += 1
            continue
        gone_transcripts.add(sid)
        transcripts_deleted += 1
        if not dry_run:
            await conn.execute(
                "DELETE FROM transcripts WHERE stream_id = ? AND tenant_id = ?",
                (sid, tenant_id),
            )
            await _tick()

    # --- Streams past the VOD window: reclaim the source (local + bucket);
    # tear down the row + dir only once no clip or transcript survives
    # under it (the FK cascade would otherwise take them down early).
    cur = await conn.execute(
        "SELECT id, source_video_path, source_audio_path, created_at FROM streams "
        "WHERE tenant_id = ?",
        (tenant_id,),
    )
    for row in await cur.fetchall():
        stream_id = str(row["id"])
        created = _parse_ts(row["created_at"])
        if created is None:
            skipped_bad_ts += 1
            _log.warning("retention.bad_timestamp", table="streams", id=stream_id)
            continue
        if created >= vod_cutoff:
            continue
        if stream_id in in_flight_streams:
            skipped_in_flight += 1
            continue
        surviving = any(
            c not in gone_clips for c in clips_by_stream.get(stream_id, [])
        ) or (stream_id in transcript_streams and stream_id not in gone_transcripts)
        if dry_run:
            vods_deleted += 1
            if not surviving:
                streams_deleted += 1
            continue
        if store_broken:
            failed += 1
            continue
        source_objects = 0
        if artifact_store is not None:
            try:
                source_objects = await _delete_prefixes(
                    artifact_store, stream_prefixes(tenant_id, stream_id)
                )
            except Exception as e:
                failed += 1
                _log.warning(
                    "retention.stream_objects_failed",
                    tenant_id=tenant_id, stream_id=stream_id, error=str(e),
                )
                continue
        video = Path(row["source_video_path"]) if row["source_video_path"] else None
        audio = Path(row["source_audio_path"]) if row["source_audio_path"] else None
        stream_dir = output_dir / stream_id
        source_bytes = reclaim_stream_source(
            stream_dir=stream_dir, source_video_path=video, source_audio_path=audio,
        )
        objects_deleted += source_objects
        bytes_freed += source_bytes
        # A stream kept for its surviving clips is revisited every run; only
        # count it when this run actually removed its source.
        if source_objects or source_bytes:
            vods_deleted += 1
        if not surviving:
            bytes_freed += _safe_unlink(stream_dir)
            await conn.execute(
                "DELETE FROM streams WHERE id = ? AND tenant_id = ?",
                (stream_id, tenant_id),
            )
            streams_deleted += 1
            await _tick()

    if not dry_run:
        await conn.commit()

    return RetentionReport(
        tenant_id=tenant_id,
        policy=policy,
        vods_deleted=vods_deleted,
        clips_deleted=clips_deleted,
        transcripts_deleted=transcripts_deleted,
        bytes_freed=bytes_freed,
        dry_run=dry_run,
        streams_deleted=streams_deleted,
        objects_deleted=objects_deleted,
        skipped_in_flight=skipped_in_flight,
        skipped_bad_timestamp=skipped_bad_ts,
        failed=failed,
    )
