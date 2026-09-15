"""Retention sweeper — hard-deletes artifacts past their tenant windows.

Three retention scopes:

  * **VOD** (source video + audio): the heaviest artifact. ~5-50 GB per
    4hr stream. Default 7 days. Normally the source is already gone — the
    pipeline deletes it on completion (see `delete_source_on_completion`);
    this window is the backstop for streams whose pipeline crashed before
    that cleanup ran.
  * **Clip** (rendered vertical clip + thumbnails): ~50-500 MB per clip.
    Default 90 days.
  * **Transcript** (Whisper JSON segments): KBs per stream. Default 365
    days — cheap to keep, useful for re-running detection later with a
    new ruleset.

Each scope is independent. A stream whose VOD aged out can still have
its transcript on disk if the transcript window hasn't passed.

Sweeper algorithm:

    1. For each tenant, resolve the three windows (col → system default).
    2. Compute three cutoff timestamps (now - window_days, UTC).
    3. SELECT streams older than VOD cutoff → unlink source files.
    4. SELECT clips older than clip cutoff → unlink clip + thumbnail files,
       DELETE the clip rows.
    5. SELECT transcripts older than transcript cutoff → DELETE rows
       (no on-disk file — transcripts live in the DB).
    6. After VOD-side artifacts are unlinked, DELETE the stream rows.
       The FK cascade nukes orphan candidates / events / etc. that
       wouldn't have been useful anyway.

`dry_run=True` walks the same scan but logs the plan instead of acting.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import os
import shutil
import time
from pathlib import Path
from typing import NamedTuple

import structlog

from chalybclip.db import Database, TenantsRepo
from chalybclip.tenancy import bound_tenant

_log = structlog.get_logger(__name__)

DEFAULT_RETENTION_VOD_DAYS = 7
DEFAULT_RETENTION_CLIP_DAYS = 90
DEFAULT_RETENTION_TRANSCRIPT_DAYS = 365


class RetentionPolicy(NamedTuple):
    """Resolved per-tenant window. NULL columns folded into system defaults."""

    vod_days: int
    clip_days: int
    transcript_days: int

    @classmethod
    def from_tenant(
        cls,
        *,
        vod_days: int | None,
        clip_days: int | None,
        transcript_days: int | None,
    ) -> RetentionPolicy:
        return cls(
            vod_days=(
                vod_days if vod_days is not None else DEFAULT_RETENTION_VOD_DAYS
            ),
            clip_days=(
                clip_days if clip_days is not None else DEFAULT_RETENTION_CLIP_DAYS
            ),
            transcript_days=(
                transcript_days
                if transcript_days is not None
                else DEFAULT_RETENTION_TRANSCRIPT_DAYS
            ),
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


# ---- helpers ----


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _iso_cutoff(days_back: int) -> str:
    """ISO 8601 UTC timestamp `days_back` ago. Matches the format
    stored in `created_at` columns so SQLite string comparison works."""
    return (_utc_now() - _dt.timedelta(days=days_back)).isoformat()


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
) -> list[RetentionReport]:
    """Sweep retention windows for every tenant (or one, when filtered).

    Args:
        db: Database handle. Migrations must already be applied.
        output_dir: Project output root (`./out`). Per-stream dirs live
            at `<output_dir>/<stream_id>/`.
        tenant_id: When set, restricts the sweep to that tenant. None
            (default) sweeps every tenant in the system.
        dry_run: When True, the report records what would be deleted but
            no DB rows or files are touched.

    Returns:
        One RetentionReport per tenant scanned. Empty list iff there are
        no tenants matching the filter.
    """
    output_dir = Path(output_dir).resolve()
    tenants_repo = TenantsRepo(db)
    if tenant_id is not None:
        t = await tenants_repo.get(tenant_id)
        tenants = [t] if t is not None else []
    else:
        tenants = await tenants_repo.list_all()

    reports: list[RetentionReport] = []
    for t in tenants:
        policy = RetentionPolicy.from_tenant(
            vod_days=t.retention_vod_days,
            clip_days=t.retention_clip_days,
            transcript_days=t.retention_transcript_days,
        )
        with bound_tenant(t.id):
            report = await _sweep_one_tenant(
                db=db,
                tenant_id=t.id,
                output_dir=output_dir,
                policy=policy,
                dry_run=dry_run,
            )
        reports.append(report)
        _log.info(
            "retention.sweep_done",
            tenant_id=t.id,
            vods_deleted=report.vods_deleted,
            clips_deleted=report.clips_deleted,
            transcripts_deleted=report.transcripts_deleted,
            bytes_freed=report.bytes_freed,
            dry_run=dry_run,
        )
    return reports


async def _sweep_one_tenant(
    *,
    db: Database,
    tenant_id: str,
    output_dir: Path,
    policy: RetentionPolicy,
    dry_run: bool,
) -> RetentionReport:
    """The per-tenant body of `sweep_retention`. Tenant is already bound."""
    vod_cutoff = _iso_cutoff(policy.vod_days)
    clip_cutoff = _iso_cutoff(policy.clip_days)
    transcript_cutoff = _iso_cutoff(policy.transcript_days)
    conn = await db.connect()

    bytes_freed = 0
    vods_deleted = 0
    clips_deleted = 0
    transcripts_deleted = 0

    # --- Clips first (FK depends on streams; deleting streams cascades).
    cur = await conn.execute(
        "SELECT id, stream_id, path, thumbnail_frame_path FROM clips "
        "WHERE tenant_id = ? AND created_at < ?",
        (tenant_id, clip_cutoff),
    )
    clip_rows = await cur.fetchall()
    # Phase 2a — the bucket mirrors the clip row's lifecycle: when the row
    # goes, its object-store copies (clip.mp4 / thumbnail / publish render)
    # go too, or the bucket accumulates unreachable objects forever.
    artifact_store = None
    if clip_rows and not dry_run:
        try:
            from chalybclip.integrations.storage import build_artifact_store
            from chalybclip.settings import get_settings

            artifact_store = build_artifact_store(get_settings())
        except Exception as e:  # a bad endpoint config must not stop the sweep
            _log.warning(
                "retention.artifact_store_unavailable",
                tenant_id=tenant_id,
                error=str(e),
            )
            artifact_store = None
    for row in clip_rows:
        clip_id = row["id"]
        clip_path = Path(row["path"]) if row["path"] else None
        clips_deleted += 1
        # Unlink the whole clip dir — that captures the mp4, thumbnail,
        # branded variants, metadata.json in one shot.
        if clip_path and not dry_run:
            bytes_freed += _safe_unlink(clip_path.parent)
        if artifact_store is not None:
            from chalybclip.integrations.storage import clip_key_family

            for key in clip_key_family(tenant_id, clip_id):
                # store.delete is best-effort (suppresses errors); a failed
                # bucket delete never blocks the row delete.
                await artifact_store.delete(key=key)
        if not dry_run:
            await conn.execute(
                "DELETE FROM clips WHERE id = ? AND tenant_id = ?",
                (clip_id, tenant_id),
            )

    # --- Transcripts (row-only — no on-disk file lives here).
    cur = await conn.execute(
        "SELECT stream_id FROM transcripts "
        "WHERE tenant_id = ? AND created_at < ?",
        (tenant_id, transcript_cutoff),
    )
    transcript_rows = list(await cur.fetchall())
    transcripts_deleted = len(transcript_rows)
    if not dry_run and transcripts_deleted:
        await conn.execute(
            "DELETE FROM transcripts "
            "WHERE tenant_id = ? AND created_at < ?",
            (tenant_id, transcript_cutoff),
        )

    # --- Streams past the VOD window: reclaim the raw source (the heavy
    # artifact). The stream ROW + the rest of the per-stream dir are only
    # removed once nothing with a LONGER window survives under them.
    #
    # This ordering matters: clips (90d) and transcripts (365d) outlive the
    # VOD window (30d), and BOTH the stream dir (`clips/` lives inside it)
    # and the stream row (FK-cascades to clip + transcript rows) would take
    # those still-in-window artifacts down early if we nuked them at the VOD
    # cutoff. So at the VOD window we delete ONLY the source; full teardown
    # waits until the stream has no surviving clips or transcripts.
    cur = await conn.execute(
        "SELECT id, source_video_path, source_audio_path FROM streams "
        "WHERE tenant_id = ? AND created_at < ?",
        (tenant_id, vod_cutoff),
    )
    from chalybclip.jobs.active import active_stream_ids

    stream_rows = await cur.fetchall()
    in_flight = active_stream_ids()
    for row in stream_rows:
        stream_id = row["id"]
        # A 30-day-old stream can still have a run in flight RIGHT NOW
        # (an operator re-running an old VOD) — leave it for the next
        # sweep rather than delete the source mid-run.
        if stream_id in in_flight:
            continue
        video = Path(row["source_video_path"]) if row["source_video_path"] else None
        audio = Path(row["source_audio_path"]) if row["source_audio_path"] else None
        stream_dir = output_dir / stream_id
        vods_deleted += 1
        if dry_run:
            continue

        # Always reclaim the source at the VOD window — usually already gone
        # (deleted on pipeline completion), so this is a cheap backstop for
        # streams whose pipeline crashed before that cleanup ran.
        bytes_freed += reclaim_stream_source(
            stream_dir=stream_dir,
            source_video_path=video,
            source_audio_path=audio,
        )

        # Surviving = newer than its own cutoff (i.e. NOT swept above). We
        # query explicitly rather than relying on the earlier DELETEs so the
        # gate is correct in dry-run too.
        cur_c = await conn.execute(
            "SELECT COUNT(*) AS n FROM clips "
            "WHERE stream_id = ? AND tenant_id = ? AND created_at >= ?",
            (stream_id, tenant_id, clip_cutoff),
        )
        surviving_clips = int((await cur_c.fetchone())["n"])
        cur_t = await conn.execute(
            "SELECT COUNT(*) AS n FROM transcripts "
            "WHERE stream_id = ? AND tenant_id = ? AND created_at >= ?",
            (stream_id, tenant_id, transcript_cutoff),
        )
        surviving_transcripts = int((await cur_t.fetchone())["n"])

        if surviving_clips == 0 and surviving_transcripts == 0:
            # Nothing longer-lived remains — tear down the rest of the dir
            # (manifests, visual signals, chat replay, diarization) and the
            # row. The FK cascade now only nukes already-aged-out rows.
            bytes_freed += _safe_unlink(stream_dir)
            await conn.execute(
                "DELETE FROM streams WHERE id = ? AND tenant_id = ?",
                (stream_id, tenant_id),
            )

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
    )
