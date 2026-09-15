"""Auto-trim a clip around its integrity issues — slice G.4b.

Two operations:

  * auto_trim_around_integrity(clip_id) — find the longest contiguous
    "clean" sub-window of the clip (no freeze/silence overlap), re-cut
    the source video into a new MP4, replace the clip's bounds, stash
    the originals so revert works.

  * revert_trim(clip_id) — restore the clip to its original bounds.
    Re-cuts the source video at the originals, replaces the clip's
    bounds + path, clears the originals column. Idempotent: a clip
    that's never been trimmed (originals NULL) is a no-op.

Both functions operate per-clip. They re-use the same ffmpeg helpers
the pipeline uses (`_ffmpeg_fast_cut` + `_ffmpeg_reformat_9_16`) so
the trimmed MP4 has identical encoding to the original.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from chalybclip.db import ClipsRepo, Database, StreamsRepo
from chalybclip.errors import ClipError

_log = structlog.get_logger(__name__)


# ---- public entry points ----


async def auto_trim_around_integrity(
    db: Database,
    *,
    clip_id: str,
    integrity_issues: list[dict[str, object]],
    min_window_s: float = 5.0,
) -> dict[str, object]:
    """Trim `clip_id` to the longest contiguous clean window.

    Args:
        db: opened DB handle, tenant already bound.
        clip_id: target clip.
        integrity_issues: list of {kind, start_s, end_s, label} dicts
            from ClipBreakdown.integrity_issues. Timestamps are in
            SOURCE-stream coordinates (same coord system as the clip
            row's start_s/end_s).
        min_window_s: refuse to trim down to less than this many
            seconds — a too-short clip won't carry a moment. Returns
            outcome="no_clean_window" in that case.

    Returns a dict with:
        outcome: "trimmed" | "no_issues" | "no_clean_window" | "noop"
        new_start_s / new_end_s / new_duration_s: when outcome=="trimmed"
        kept_label: human-readable explanation
    """
    clips_repo = ClipsRepo(db)
    streams_repo = StreamsRepo(db)

    clip = await clips_repo.get(clip_id)
    if clip is None:
        raise ClipError(f"clip {clip_id!r} not found")

    if not integrity_issues:
        return {"outcome": "no_issues"}

    # Translate each issue's source timestamps to clip-local time
    # (relative to clip.start_s), clip to [0, clip.duration_s].
    bad_ranges: list[tuple[float, float]] = []
    for issue in integrity_issues:
        s = float(issue.get("start_s", 0.0))
        e = float(issue.get("end_s", 0.0))
        local_s = max(0.0, s - clip.start_s)
        local_e = min(clip.duration_s, e - clip.start_s)
        if local_e > local_s:
            bad_ranges.append((local_s, local_e))
    if not bad_ranges:
        return {"outcome": "no_issues"}

    # Find the longest clean window between [0, duration_s] minus the
    # union of bad_ranges. We sort + merge bad_ranges first so simple
    # gap-walk works.
    bad_ranges.sort()
    merged: list[tuple[float, float]] = []
    for r in bad_ranges:
        if merged and r[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r[1]))
        else:
            merged.append(r)

    candidates: list[tuple[float, float]] = []  # local clip coords
    cursor = 0.0
    for s, e in merged:
        if s > cursor:
            candidates.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < clip.duration_s:
        candidates.append((cursor, clip.duration_s))

    if not candidates:
        return {"outcome": "no_clean_window"}

    best = max(candidates, key=lambda r: r[1] - r[0])
    best_len = best[1] - best[0]
    if best_len < min_window_s:
        return {"outcome": "no_clean_window"}

    # Translate back to source-stream coords.
    new_start_s = clip.start_s + best[0]
    new_end_s = clip.start_s + best[1]
    new_duration_s = new_end_s - new_start_s

    # Re-cut the MP4. Prefer the source VOD (full re-cut + 9:16 reformat).
    # When the VOD has been pruned by the retention sweeper, fall back to
    # slicing the clean sub-window straight out of the EXISTING clip MP4:
    # it's already 9:16, and `best` is a sub-window of the clip in
    # clip-local coords, so a plain fast-cut yields the same window without
    # the source. This keeps auto-trim working on retention-pruned streams.
    stream = await streams_repo.get(clip.stream_id)
    source_video = Path(stream.source_video_path) if stream is not None else None

    # First trim: stash a pristine copy of the clip MP4 next to it so revert
    # can always restore it — even after the source VOD is pruned. We do this
    # BEFORE overwriting clip.mp4 below. Subsequent trims keep the original
    # copy (the FIRST pre-trim state), matching how original_start_s is kept.
    if clip.original_start_s is None:
        _save_pretrim_original(Path(clip.path))

    if source_video is not None and source_video.exists():
        new_path = _recut_clip_mp4(
            source_video=source_video,
            out_path=Path(clip.path),
            start_s=new_start_s,
            duration_s=new_duration_s,
        )
    else:
        clip_mp4 = Path(clip.path)
        if not clip_mp4.exists():
            raise ClipError(
                f"source video missing AND clip MP4 missing ({clip_mp4}); "
                f"re-ingest the stream to trim its clips."
            )
        # `best` is clip-local — the clip MP4 timeline starts at 0 == the
        # clip's current start_s, so best[0] indexes straight into it.
        new_path = _recut_from_existing_clip(
            clip_mp4=clip_mp4,
            local_start_s=best[0],
            duration_s=best_len,
        )

    # First trim: stash originals. Subsequent trim (operator clicks
    # auto-trim twice): keep the FIRST original_start_s/end_s.
    original_start_s = (
        clip.original_start_s if clip.original_start_s is not None else clip.start_s
    )
    original_end_s = (
        clip.original_end_s if clip.original_end_s is not None else clip.end_s
    )
    await clips_repo.set_trim_bounds(
        clip_id,
        start_s=new_start_s,
        end_s=new_end_s,
        duration_s=new_duration_s,
        new_path=str(new_path),
        original_start_s=original_start_s,
        original_end_s=original_end_s,
    )

    # Invalidate cached MP4 exports so the next download lazy-regenerates
    # with the new bounds.
    _invalidate_export_cache(Path(clip.path).parent)

    _log.info(
        "clip.auto_trim",
        clip_id=clip_id,
        old_bounds=(clip.start_s, clip.end_s),
        new_bounds=(new_start_s, new_end_s),
        issues_skipped=len(merged),
    )
    return {
        "outcome": "trimmed",
        "new_start_s": new_start_s,
        "new_end_s": new_end_s,
        "new_duration_s": new_duration_s,
        "kept_label": (
            f"Kept the {int(best_len)}s clean window from "
            f"{int(new_start_s)}-{int(new_end_s)}s; skipped "
            f"{len(merged)} integrity issue(s)."
        ),
    }


async def revert_trim(
    db: Database,
    *,
    clip_id: str,
) -> dict[str, object]:
    """Restore `clip_id` to its original bounds.

    No-op when the clip has never been trimmed (original_start_s
    is NULL). Otherwise restore the pristine pre-trim MP4 copy stashed
    on the first trim (`clip.original.mp4`) — that works even after the
    source VOD is pruned. Only when that copy is missing do we fall back
    to re-cutting the source video at the originals.
    """
    clips_repo = ClipsRepo(db)
    streams_repo = StreamsRepo(db)

    clip = await clips_repo.get(clip_id)
    if clip is None:
        raise ClipError(f"clip {clip_id!r} not found")

    if clip.original_start_s is None or clip.original_end_s is None:
        return {"outcome": "noop"}

    original_start = float(clip.original_start_s)
    original_end = float(clip.original_end_s)
    original_duration = original_end - original_start

    clip_mp4 = Path(clip.path)
    pretrim = _pretrim_original_path(clip_mp4)
    if pretrim.exists():
        # Fast path: swap the pristine copy back into place. No source
        # video, no ffmpeg re-encode — byte-identical to the original cut.
        pretrim.replace(clip_mp4)
        new_path = clip_mp4
    else:
        stream = await streams_repo.get(clip.stream_id)
        if stream is None:
            raise ClipError(f"stream {clip.stream_id!r} not found for clip revert")
        source_video = Path(stream.source_video_path)
        if not source_video.exists():
            raise ClipError(
                f"no pre-trim copy and source video missing: {source_video}. "
                f"Can't revert without re-cutting; re-ingest the stream to "
                f"restore the original bounds for its clips."
            )
        new_path = _recut_clip_mp4(
            source_video=source_video,
            out_path=clip_mp4,
            start_s=original_start,
            duration_s=original_duration,
        )

    # Restore the bounds + path; pass originals=None so set_trim_bounds
    # clears them in the same query.
    await clips_repo.set_trim_bounds(
        clip_id,
        start_s=original_start,
        end_s=original_end,
        duration_s=original_duration,
        new_path=str(new_path),
        original_start_s=None,
        original_end_s=None,
    )
    _invalidate_export_cache(Path(clip.path).parent)

    _log.info(
        "clip.revert_trim",
        clip_id=clip_id,
        restored_bounds=(original_start, original_end),
    )
    return {
        "outcome": "reverted",
        "new_start_s": original_start,
        "new_end_s": original_end,
        "new_duration_s": original_duration,
    }


# ---- helpers ----


def _recut_clip_mp4(
    *,
    source_video: Path,
    out_path: Path,
    start_s: float,
    duration_s: float,
) -> Path:
    """Re-cut the clip MP4 from `source_video` with the new bounds.

    Goes via a temp intermediate then the same 9:16 reformat as the
    initial cut pipeline so encoding/format stays identical. Replaces
    the file at `out_path` atomically (write to .tmp, rename).
    """
    # Import locally to avoid a service.py → trim.py → service.py cycle
    # at module load time. service.py imports from clip.__init__ which
    # re-exports trim symbols.
    from .service import _ffmpeg_fast_cut, _ffmpeg_reformat_9_16, _safe_analyze_framing
    from chalybclip.config import ClipConfig

    cfg = ClipConfig()
    intermediate = out_path.parent / f"{out_path.stem}.trim.intermediate.mp4"
    tmp_out = out_path.with_suffix(".trim.tmp.mp4")
    # Slice G.4c — re-run framing detection on the new window so a horizontal
    # source still gets the blurred-letterbox path on trim/revert.
    framing_verdict = _safe_analyze_framing(
        video_path=source_video, start_s=start_s, end_s=start_s + duration_s
    )
    try:
        _ffmpeg_fast_cut(
            video_path=source_video,
            start_s=start_s,
            duration_s=duration_s,
            out_path=intermediate,
        )
        _ffmpeg_reformat_9_16(
            in_path=intermediate,
            out_path=tmp_out,
            cfg=cfg,
            smart_box=None,  # smart_box was computed for the original
                             # window; recomputing per-trim is overkill
                             # for the MVP. Center-crop fallback applies.
            brand_kit=None,  # brand_kit overlay is burned at download
                             # time via the recorder/burn path, not in
                             # the source MP4.
            framing_verdict=framing_verdict,
        )
        # Replace atomically.
        tmp_out.replace(out_path)
    finally:
        if intermediate.exists():
            intermediate.unlink()
        if tmp_out.exists():
            tmp_out.unlink()
    return out_path


def _pretrim_original_path(clip_mp4: Path) -> Path:
    """Path of the pristine pre-trim copy stashed beside the clip MP4."""
    return clip_mp4.with_name(f"{clip_mp4.stem}.original{clip_mp4.suffix}")


def _save_pretrim_original(clip_mp4: Path) -> None:
    """Copy the current clip MP4 to its pre-trim sidecar, once.

    Best-effort: a missing clip MP4 (already gone) or a copy failure must
    not block the trim itself — revert just falls back to the source VOD.
    Never overwrites an existing copy, so repeated trims keep the FIRST
    pristine state.
    """
    import shutil

    dest = _pretrim_original_path(clip_mp4)
    if dest.exists() or not clip_mp4.exists():
        return
    try:
        shutil.copy2(clip_mp4, dest)
    except Exception:  # noqa: BLE001 — revert degrades to source re-cut
        pass


def _recut_from_existing_clip(
    *,
    clip_mp4: Path,
    local_start_s: float,
    duration_s: float,
) -> Path:
    """Fallback re-cut when the source VOD is gone: slice the clean
    sub-window out of the EXISTING clip MP4.

    The clip MP4 is already 9:16-reformatted, so we only fast-cut — no
    reformat pass, no source video needed. `local_start_s` is relative to
    the clip's current start (the clip MP4 timeline starts at 0). Reads
    from `clip_mp4` into a separate temp file, then replaces it atomically.

    Note: this is sample-accurate but re-encodes the already-encoded clip,
    so it incurs one generation of recompression. revert_trim still needs
    the source VOD — the trimmed-away footage isn't recoverable from this.
    """
    from .service import _ffmpeg_fast_cut
    from chalybclip.config import ClipConfig

    cfg = ClipConfig()
    tmp_out = clip_mp4.with_suffix(".trim.tmp.mp4")
    try:
        _ffmpeg_fast_cut(
            video_path=clip_mp4,
            start_s=local_start_s,
            duration_s=duration_s,
            out_path=tmp_out,
            cfg=cfg,
        )
        tmp_out.replace(clip_mp4)
    finally:
        if tmp_out.exists():
            tmp_out.unlink()
    return clip_mp4


def _invalidate_export_cache(clip_dir: Path) -> None:
    """Nuke every cached export so the next download lazy-regenerates
    with the new bounds. Slice J.2a added per-resolution variants; we
    invalidate them all here because the source MP4 changed and any
    pre-existing 4K cache is stale too."""
    names = (
        "clip_render.mp4",
        "clip_render_1080.mp4",
        "clip_render_2k.mp4",
        "clip_render_4k.mp4",
        "clip_final.mp4",
    )
    for name in names:
        path = clip_dir / name
        try:
            if path.exists():
                path.unlink()
        except Exception:  # noqa: BLE001 — best-effort
            pass
