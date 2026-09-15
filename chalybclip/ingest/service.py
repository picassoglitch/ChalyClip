"""VOD ingest: download via yt-dlp, extract a 16 kHz mono WAV via ffmpeg.

The output layout for a stream is:

    <output_dir>/<stream_id>/
        stream.json                # serialized Stream model
        source/
            video.mp4              # downloaded VOD (remuxed to mp4)
            audio.wav              # 16 kHz mono PCM extract

Idempotency: if `stream.json` exists, the function returns the saved Stream
without re-downloading or re-extracting unless `force=True`.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import yt_dlp

from chalybclip.errors import IngestError
from chalybclip.events import (
    STREAM_AUDIO_EXTRACTED,
    STREAM_DOWNLOAD_COMPLETED,
    STREAM_DOWNLOAD_STARTED,
    emit,
)
from chalybclip.ids import new_id

if TYPE_CHECKING:
    from chalybclip.db import Database

from .models import Platform, Stream

_log = structlog.get_logger(__name__)

_PLATFORM_PATTERNS: dict[Platform, re.Pattern[str]] = {
    "kick": re.compile(r"(?:^|//|\.)kick\.com/", re.IGNORECASE),
    "twitch": re.compile(r"(?:^|//|\.)twitch\.tv/", re.IGNORECASE),
    "youtube": re.compile(r"(?:^|//|\.)(?:youtube\.com|youtu\.be)/", re.IGNORECASE),
}


def detect_platform(vod_url: str) -> Platform:
    """Map a VOD URL to a known platform tag (or `unknown`)."""
    for platform, pattern in _PLATFORM_PATTERNS.items():
        if pattern.search(vod_url):
            return platform
    return "unknown"


def is_ffmpeg_available() -> bool:
    """True iff `ffmpeg` is resolvable on PATH.

    Used by the upload endpoints to fail fast: a 6 GB upload that 400s on
    audio extraction is a bad first impression. The dashboard reads this at
    page-render time too, so the upload form can disable itself with a clear
    install hint instead of accepting a doomed POST.
    """
    return shutil.which("ffmpeg") is not None


async def ingest_vod(
    tenant_id: str,
    vod_url: str,
    output_dir: Path,
    *,
    stream_id: str | None = None,
    force: bool = False,
    chat_replay_source: Path | None = None,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
    db: "Database | None" = None,
) -> Stream:
    """Download a VOD and extract its audio. Returns a Stream.

    Args:
        tenant_id: Tenant owning the stream. Phase 0 hardcodes `"default"`.
        vod_url: VOD URL on Kick / Twitch / YouTube.
        output_dir: Root directory; a `<stream_id>/` subdirectory is created.
        stream_id: Optional existing stream ID for resumes. New ULID if omitted.
        force: If true, overwrite existing files even when `stream.json` exists.
        chat_replay_source: Optional path to a JSONL of chat messages
            (one `ChatMessage` per line). Phase 1 doesn't fetch chat replay
            from platforms automatically - pass a pre-fetched file or skip.
        cookies_from_browser: Pass cookies from a logged-in browser session
            ("chrome" / "edge" / "firefox" / "brave" / "chromium") through to
            yt-dlp. Required for Kick. Falls back to `Settings.cookies_from_browser`
            when omitted. Conflicts with `cookies_file`; if both are set,
            `cookies_file` wins.
        cookies_file: Alternative to `cookies_from_browser` — absolute path
            to a Netscape-format cookies.txt file (exported via a browser
            extension). Browser can stay open. Falls back to
            `Settings.cookies_file` when omitted.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sid = stream_id or new_id("str")
    stream_dir = output_dir / sid
    source_dir = stream_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    stream_json = stream_dir / "stream.json"
    video_path = source_dir / "video.mp4"
    audio_path = source_dir / "audio.wav"

    if not force and stream_json.exists():
        # Cache-resume is only valid while the ARTIFACTS the pipeline needs
        # are still on disk. The source gets reclaimed out from under the
        # manifest by design — delete-on-completion, the failure-path
        # cleanup in the pipeline runner, the disk watchdog — and every one
        # of those relies on "a re-run re-downloads automatically". Resuming
        # on stream.json alone broke that contract: the re-run skipped the
        # download and then died at transcribe on the missing audio.
        if video_path.exists() and audio_path.exists():
            cached = Stream.model_validate_json(stream_json.read_text("utf-8"))
            # Resume: still import chat replay if a new source is provided
            # this run.
            if chat_replay_source is not None:
                from .chat_replay import chat_replay_path, import_chat_replay

                import_chat_replay(
                    source=chat_replay_source,
                    stream_dir=stream_dir,
                    stream_id=cached.id,
                    tenant_id=cached.tenant_id,
                )
                cached = cached.model_copy(
                    update={"source_chat_path": chat_replay_path(stream_dir)}
                )
                stream_json.write_text(
                    cached.model_dump_json(indent=2), encoding="utf-8"
                )
            return cached
        # Uploads and live recordings have no origin to re-fetch from —
        # falling through to yt-dlp on their pseudo-URLs would produce an
        # opaque "unsupported URL" error. Say what actually happened.
        if vod_url.startswith(("upload://", "live://")):
            raise IngestError(
                f"source for {sid} was reclaimed and {vod_url!r} has no "
                "origin to re-download from — upload the file again to "
                "re-process it"
            )
        _log.info(
            "ingest.resume.source_missing",
            stream_id=sid,
            reason="stream.json cached but source video/audio reclaimed — "
                   "re-downloading",
        )

    platform = detect_platform(vod_url)

    # Resolve cookie auth: explicit kwargs win; otherwise fall back to
    # Settings (env vars / .env). cookies_file takes precedence over
    # cookies_from_browser when both are set, since the file path is the
    # workaround for the "Chrome holds the cookie DB lock" failure mode.
    if cookies_file is None or cookies_from_browser is None:
        from chalybclip.settings import get_settings

        settings = get_settings()
        if cookies_file is None:
            cookies_file = settings.cookies_file or None
        if cookies_from_browser is None:
            cookies_from_browser = settings.cookies_from_browser or None

    # Free-disk preflight, self-healing: if we're low, shed the oldest raw
    # sources FIRST (the disk budget), then fail fast only if still below the
    # hard floor. Prevents a near-full volume from blocking new ingests when
    # there are stale sources sitting around to reclaim.
    if db is not None:
        from chalybclip.settings import get_settings as _gs

        floor = int(getattr(_gs(), "min_free_disk_bytes", 0) or 0)
        if floor > 0:
            try:
                free = shutil.disk_usage(output_dir).free
            except OSError:
                free = floor  # can't stat → don't try to reclaim
            if free < floor:
                from chalybclip.retention import reclaim_sources_until_free

                with contextlib.suppress(Exception):  # reclaim is best-effort
                    await reclaim_sources_until_free(
                        db, output_dir=output_dir, target_free_bytes=floor * 2,
                    )
    _assert_disk_headroom(output_dir)

    # Task 2a — telemetry. Wall-clock + file-size + downloader-id around
    # both the yt-dlp call and the audio extract so the dashboard can
    # break the opaque "ingesting" row into download / extract phases
    # and the operator can SEE whether a slow ingest is bandwidth-bound
    # (download MB and Mbps in the payload) or CPU-bound (audio extract
    # ate 12 minutes on a multi-hour VOD).
    await emit(
        db,
        STREAM_DOWNLOAD_STARTED,
        {
            "stream_id": sid,
            "vod_url": vod_url,
            "platform": platform,
        },
    )
    download_t0 = time.monotonic()
    info = await asyncio.to_thread(
        _download_vod,
        vod_url=vod_url,
        target_path=video_path,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
        platform=platform,
    )
    download_wall_s = time.monotonic() - download_t0

    # File-size + throughput AFTER the download — we trust the on-disk
    # file because yt-dlp's `filesize_approx` field can be off by 20%+
    # for HLS sources where the segment-count × avg-segment-bytes is
    # estimated rather than measured.
    file_bytes = video_path.stat().st_size if video_path.exists() else 0
    throughput_mbps = (
        (file_bytes * 8.0 / 1e6) / download_wall_s
        if download_wall_s > 0 and file_bytes > 0
        else 0.0
    )
    chosen_format = info.get("format_id") or info.get("format") or "unknown"
    # yt-dlp picks a "protocol" string per format: hls / m3u8 / m3u8_native /
    # http / https / dash / etc. Useful in the payload because that's the
    # signal that tells us whether `concurrent_fragment_downloads` even
    # applies (it only matters for segmented protocols).
    chosen_protocol = info.get("protocol") or "unknown"
    _log.info(
        "ingest.download.complete",
        stream_id=sid,
        vod_url=vod_url,
        platform=platform,
        wall_s=round(download_wall_s, 2),
        file_bytes=file_bytes,
        throughput_mbps=round(throughput_mbps, 1),
        format_id=chosen_format,
        protocol=chosen_protocol,
    )
    await emit(
        db,
        STREAM_DOWNLOAD_COMPLETED,
        {
            "stream_id": sid,
            "wall_s": round(download_wall_s, 2),
            "file_bytes": file_bytes,
            "throughput_mbps": round(throughput_mbps, 1),
            "format_id": chosen_format,
            "protocol": chosen_protocol,
        },
    )

    if force or not audio_path.exists():
        extract_t0 = time.monotonic()
        await asyncio.to_thread(_extract_audio, video_path, audio_path)
        extract_wall_s = time.monotonic() - extract_t0
        _log.info(
            "ingest.audio_extract.complete",
            stream_id=sid, wall_s=round(extract_wall_s, 2),
        )
        await emit(
            db,
            STREAM_AUDIO_EXTRACTED,
            {
                "stream_id": sid,
                "wall_s": round(extract_wall_s, 2),
            },
        )

    # Duration: ffprobe the file we ACTUALLY downloaded — don't trust
    # yt-dlp's metadata. A freshly-uploaded YouTube video is often still
    # being processed server-side: yt-dlp can pull a truncated progressive
    # stream (format 18) while the info dict still advertises the FULL
    # length. Trusting the metadata there sizes the whole pipeline for a
    # 32-min video that's a few seconds on disk, and detection silently
    # emits one throwaway clip from the start. ffprobe of the on-disk file
    # is authoritative; metadata is only the fallback when ffprobe can't read.
    claimed_s = float(info.get("duration") or 0.0)
    actual_s = await asyncio.to_thread(_ffprobe_duration, video_path)
    _assert_download_complete(
        vod_url=vod_url, claimed_s=claimed_s, actual_s=actual_s
    )
    duration_s = actual_s if actual_s > 0.0 else claimed_s

    chat_path: Path | None = None
    if chat_replay_source is not None:
        from .chat_replay import chat_replay_path, import_chat_replay

        import_chat_replay(
            source=chat_replay_source,
            stream_dir=stream_dir,
            stream_id=sid,
            tenant_id=tenant_id,
        )
        chat_path = chat_replay_path(stream_dir)

    stream = Stream(
        id=sid,
        tenant_id=tenant_id,
        vod_url=vod_url,
        platform=platform,
        title=info.get("title"),
        channel=info.get("uploader") or info.get("channel"),
        duration_s=duration_s,
        source_video_path=video_path,
        source_audio_path=audio_path,
        source_chat_path=chat_path,
    )
    stream_json.write_text(stream.model_dump_json(indent=2), encoding="utf-8")
    return stream


async def ingest_uploaded(
    tenant_id: str,
    source_path: Path,
    output_dir: Path,
    *,
    stream_id: str | None = None,
    title: str | None = None,
    chat_replay_source: Path | None = None,
) -> Stream:
    """Ingest a video the operator uploaded (no yt-dlp involved).

    Mirrors `ingest_vod` but skips the download step entirely. The on-disk
    layout is identical so every downstream step (transcribe, detect, cut,
    variants) finds its inputs in the usual places:

        <output_dir>/<stream_id>/
            stream.json
            source/
                video.mp4
                audio.wav

    `source_path` may be any container ffmpeg understands. We do NOT remux —
    just copy/move the file to `source/video.mp4`. ffmpeg reads it for the
    audio extract and ffprobe reads it for duration; both work on whatever
    actual container is inside.

    Args:
        tenant_id: Tenant owning the stream.
        source_path: An on-disk file the user just uploaded. The caller is
            responsible for streaming the upload to a tempfile first; this
            function moves it into place.
        output_dir: Root directory; a `<stream_id>/` subdirectory is created.
        stream_id: Optional existing stream ID for resumes. New ULID otherwise.
        title: Display title — usually the original filename.
        chat_replay_source: Same semantics as `ingest_vod`. Most uploads
            won't have one and this stays None.

    Returns:
        The persisted Stream. Idempotent on `stream_id`: a re-call with the
        same id and an existing stream.json returns the cached row.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_path).resolve()
    if not source_path.exists():
        raise IngestError(f"upload source not found: {source_path}")

    sid = stream_id or new_id("str")
    stream_dir = output_dir / sid
    source_dir = stream_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    stream_json = stream_dir / "stream.json"
    video_path = source_dir / "video.mp4"
    audio_path = source_dir / "audio.wav"

    if stream_json.exists():
        cached = Stream.model_validate_json(stream_json.read_text("utf-8"))
        if chat_replay_source is not None:
            from .chat_replay import chat_replay_path, import_chat_replay

            import_chat_replay(
                source=chat_replay_source,
                stream_dir=stream_dir,
                stream_id=cached.id,
                tenant_id=cached.tenant_id,
            )
            cached = cached.model_copy(
                update={"source_chat_path": chat_replay_path(stream_dir)}
            )
            stream_json.write_text(cached.model_dump_json(indent=2), encoding="utf-8")
        return cached

    # Move the uploaded tempfile into place. Move beats copy because the
    # uploaded file is in a temp dir we own and it's likely 100s of MB.
    if video_path.exists():
        video_path.unlink()
    try:
        source_path.replace(video_path)
    except OSError as e:
        # Cross-device move on Windows — fall back to copy.
        import shutil

        shutil.copy2(source_path, video_path)
        try:
            source_path.unlink()
        except OSError:
            pass
        del e

    await asyncio.to_thread(_extract_audio, video_path, audio_path)
    duration_s = await asyncio.to_thread(_ffprobe_duration, video_path)

    chat_path: Path | None = None
    if chat_replay_source is not None:
        from .chat_replay import chat_replay_path, import_chat_replay

        import_chat_replay(
            source=chat_replay_source,
            stream_dir=stream_dir,
            stream_id=sid,
            tenant_id=tenant_id,
        )
        chat_path = chat_replay_path(stream_dir)

    # `vod_url` is required by the Stream schema but uploads have no canonical
    # URL. Stash an `upload://` pseudo-URL so logs / dashboards don't show an
    # empty string and downstream `ingest_vod(stream_id=...)` cache-resume
    # calls have something stable to dedupe on.
    pseudo_url = f"upload://{title or video_path.name}"

    stream = Stream(
        id=sid,
        tenant_id=tenant_id,
        vod_url=pseudo_url,
        platform="upload",
        title=title,
        channel=None,
        duration_s=duration_s,
        source_video_path=video_path,
        source_audio_path=audio_path,
        source_chat_path=chat_path,
    )
    stream_json.write_text(stream.model_dump_json(indent=2), encoding="utf-8")
    return stream


def youtube_extractor_args(settings: Any) -> dict[str, list[str]] | None:
    """Build yt-dlp's `extractor_args["youtube"]` from the free bot-gate
    knobs — a player_client switch and an optional self-hosted PO-token
    provider. Both are FREE and need no proxy/cookies; YouTube-scoped, so
    they never affect other extractors. Returns None when neither is set.

    Shared by the VOD download and the channel-VOD listing so both egress
    paths dodge the gate the same way.
    """
    yt: dict[str, list[str]] = {}
    clients = (getattr(settings, "ytdlp_player_client", None) or "").strip()
    if clients:
        yt["player_client"] = [c.strip() for c in clients.split(",") if c.strip()]
    po_url = (getattr(settings, "ytdlp_po_provider_url", None) or "").strip()
    if po_url:
        # The bgutil yt-dlp plugin reads its provider base URL from here.
        yt["getpot_bgutil_baseurl"] = [po_url]
    return yt or None


def _download_vod(
    *,
    vod_url: str,
    target_path: Path,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
    platform: Platform = "unknown",
) -> dict[str, Any]:
    """Run yt-dlp; ensure the resulting file lives at exactly `target_path`.

    yt-dlp picks the file extension from the chosen format, so we let it
    write `video.<ext>` and rename to `video.mp4` once we know the actual
    path. `merge_output_format=mp4` keeps the container consistent for
    downstream ffmpeg work.

    Cookie auth (used by Kick + age-gated YouTube):
      * `cookies_file` — Netscape-format cookies.txt (browser stays open;
        most reliable on Windows where Chrome locks its cookie DB).
      * `cookies_from_browser` — pull cookies live from a browser profile.
        Faster setup but breaks if Chrome is running.
    When both are set, `cookies_file` wins.

    Task 2b — three knobs added without changing the default outcome:

      * `format` is now bounded by `Settings.max_source_height` (default
        1080). Twitch / Kick / YouTube routinely expose 4K+ on popular
        channels and the old "best available" was pulling source frames
        we'd just throw away in the 9:16 reformat. Capping the source
        at 1080 cuts the download size 3-5× without changing the output
        clip resolution.
      * `concurrent_fragment_downloads = Settings.ytdlp_concurrent_fragments`
        (default 16). HLS / DASH segmented downloads — which is
        every Twitch / Kick / YouTube VOD — fan out per-segment instead
        of single-threading the fetch. 3-10× speedup is typical.
      * `external_downloader = "aria2c"` when `Settings.ytdlp_use_aria2c`
        is True AND aria2c is on PATH. aria2c parallelises within a
        single HTTP fetch via byte-range; helpful for non-segmented
        sources (direct mp4 URLs) where `concurrent_fragment_downloads`
        is a no-op.

    Defaults preserve current behavior bit-for-bit on hosts where
    none of the new env vars are set.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    outtmpl = str(target_path.with_suffix("")) + ".%(ext)s"

    # Read settings lazily so test fixtures that monkeypatch
    # `get_settings.cache_clear` see fresh values per test.
    from chalybclip.settings import get_settings
    s = get_settings()
    max_h = int(getattr(s, "max_source_height", 1080) or 0)
    concurrent_frags = max(1, int(getattr(s, "ytdlp_concurrent_fragments", 1)))
    use_aria2c = bool(getattr(s, "ytdlp_use_aria2c", False))

    # Format ladder. The two-stage `/best` fallback keeps yt-dlp happy
    # when a platform only exposes pre-muxed formats (single track)
    # rather than separate video + audio streams.
    if max_h > 0:
        fmt = (
            f"bestvideo[height<={max_h}]*+bestaudio/"
            f"best[height<={max_h}]/best"
        )
    else:
        fmt = "bestvideo*+bestaudio/best"

    ydl_opts: dict[str, Any] = {
        "outtmpl": outtmpl,
        "format": fmt,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "writeinfojson": False,
        "writethumbnail": False,
        "overwrites": True,
        # Task 2b — concurrent_fragment_downloads is yt-dlp's name for
        # the HLS / DASH fan-out cap. Plumbed through; safe to set on
        # non-segmented formats (silently ignored by yt-dlp).
        "concurrent_fragment_downloads": concurrent_frags,
    }

    # Residential proxy — the sustainable fix for datacenter-IP bot-gating
    # (YouTube "confirm you're not a bot", Kick Cloudflare 403). Routes the
    # whole fetch through a home-looking IP so we don't depend on cookies
    # that expire. Unset → direct. See Settings.ytdlp_proxy.
    proxy = (getattr(s, "ytdlp_proxy", None) or "").strip()
    if proxy:
        ydl_opts["proxy"] = proxy

    # FREE bot-gate mitigation: a YouTube player_client switch (+ optional
    # PO-token provider) often clears the gate with NO proxy and NO cookies.
    # YouTube-scoped, so it never affects Twitch/Kick. See Settings.
    yt_args = youtube_extractor_args(s)
    if yt_args:
        ydl_opts["extractor_args"] = {"youtube": yt_args}

    # Task 2b — aria2c external downloader. Probe at request time so
    # the operator can flip the env var without restarting (the
    # `get_settings` cache survives but the PATH probe doesn't).
    # `for: default` means aria2c is used for the default group, which
    # is what yt-dlp routes the actual media segments through.
    if use_aria2c and shutil.which("aria2c"):
        ydl_opts["external_downloader"] = {"default": "aria2c"}
        ydl_opts["external_downloader_args"] = {
            "aria2c": [
                "-x16",       # 16 connections per server
                "-s16",       # split each download into 16 streams
                "-k1M",       # min split size 1 MB
                "--summary-interval=0",
            ],
        }
    elif use_aria2c:
        # Operator wanted aria2c but the binary isn't installed. Log
        # once so the symptom (no speedup) is visible.
        _log.warning(
            "ingest.aria2c_requested_but_missing",
            hint="install aria2c on the host or set CHALYBCLIP_YTDLP_USE_ARIA2C=false",
        )

    # Kick sits behind Cloudflare and 403s plain yt-dlp requests. Tell
    # yt-dlp to impersonate a browser (needs curl_cffi, a hard dep). Same
    # bypass we use to list Kick channel VODs. Best-effort: if the
    # impersonate backend is unavailable, fall through and let the normal
    # error path explain the 403 rather than crashing here.
    if platform == "kick":
        try:
            from yt_dlp.networking.impersonate import ImpersonateTarget

            ydl_opts["impersonate"] = ImpersonateTarget("chrome")
        except Exception as e:  # impersonate is best-effort
            _log.warning(
                "ingest.kick_impersonate_unavailable",
                error=str(e),
                hint="install curl_cffi so yt-dlp can bypass Kick's Cloudflare",
            )

    if cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)
    elif cookies_from_browser:
        # yt-dlp's option is a tuple: (BROWSER, [PROFILE, [KEYRING, [CONTAINER]]]).
        # We only ever pass the browser name; users with multiple profiles can
        # extend this later.
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser.strip().lower(),)

    def _fetch() -> dict[str, Any]:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(vod_url, download=True) or {}

    # Catch YoutubeDLError, not DownloadError: UnavailableVideoError (raised
    # when yt-dlp trips over its own broken resume state) is a SIBLING of
    # DownloadError, so the narrower catch let it escape raw to the pipeline
    # — no sweep, no hint, and the stale partial poisoned every re-run.
    try:
        info: dict[str, Any] = _fetch()
    except yt_dlp.utils.YoutubeDLError as e:
        # Sweep the partial download (`.part`, `.ytdl`, fragment files, and a
        # half-written muxed file) before going further. A failed fetch of a
        # multi-hour VOD otherwise leaves gigabytes behind that the next
        # download has to fit around — the start of the disk-exhaustion
        # spiral.
        _sweep_download_leftovers(target_path)
        if _is_cookie_acquisition_error(e) and "cookiesfrombrowser" in ydl_opts:
            # The browser holds an exclusive lock on its cookie DB (yt-dlp
            # #7271) — a machine-local problem, not a site refusal. Cookies
            # are usually unnecessary on residential IPs, so retry the same
            # download WITHOUT them rather than surfacing close-your-browser
            # instructions to the operator. If the site then 403s, the
            # normal explain path below still says so.
            _log.warning(
                "ingest.download.cookies_unavailable_retry_without",
                vod_url=vod_url,
                error=str(e)[:200],
            )
            ydl_opts.pop("cookiesfrombrowser", None)
        elif not _is_stale_partial_error(e):
            raise _explain_download_failure(
                err=e, vod_url=vod_url, platform=platform,
                cookies_from_browser=cookies_from_browser,
                cookies_file=cookies_file,
            ) from e
        else:
            # A killed run (host restart mid-download) leaves `.part` + `.ytdl`
            # resume state pointing at fragment files that may no longer all
            # exist; yt-dlp's resume then fails deterministically with a local
            # FileNotFoundError. The sweep above just removed that state, so
            # one clean-slate retry is the self-heal — without it the user
            # re-runs into the identical error forever.
            _log.warning(
                "ingest.download.stale_partial_retry",
                vod_url=vod_url,
                error=str(e),
            )
        try:
            info = _fetch()
        except yt_dlp.utils.YoutubeDLError as e2:
            _sweep_download_leftovers(target_path)
            raise _explain_download_failure(
                err=e2, vod_url=vod_url, platform=platform,
                cookies_from_browser=cookies_from_browser,
                cookies_file=cookies_file,
            ) from e2

    actual = _resolve_downloaded_path(info, fallback=target_path)
    if actual.resolve() != target_path.resolve():
        if target_path.exists():
            target_path.unlink()
        actual.replace(target_path)
    return info


def _assert_disk_headroom(output_dir: Path) -> None:
    """Raise IngestError when the output volume is below the configured free-
    space floor (`min_free_disk_bytes`). No-op when the floor is 0/unset or
    the volume can't be statted (don't block ingest on an observability gap).
    """
    from chalybclip.settings import get_settings

    floor = int(getattr(get_settings(), "min_free_disk_bytes", 0) or 0)
    if floor <= 0:
        return
    try:
        free = shutil.disk_usage(output_dir).free
    except OSError:
        return
    if free < floor:
        gb = 1024 * 1024 * 1024
        raise IngestError(
            f"insufficient disk: {free / gb:.1f} GB free on the output volume, "
            f"need at least {floor / gb:.1f} GB to start a download. Free space "
            "(run the retention sweep, delete old streams, or expand the volume) "
            "and retry."
        )


def _sweep_download_leftovers(target_path: Path) -> None:
    """Delete every on-disk artifact of a partial download for this stream's
    source: `video.mp4.part`, the `.ytdl` resume-state file, concurrent
    fragment pieces (`video.mp4.part-FragN`), and a half-written muxed file.
    They all share the outtmpl stem inside this stream's own `source/` dir,
    so globbing the stem is safe. Best-effort — a locked file stays behind.
    """
    stem = target_path.with_suffix("").name
    for leftover in target_path.parent.glob(f"{stem}*"):
        with contextlib.suppress(OSError):
            if leftover.is_file():
                leftover.unlink()


def _is_cookie_acquisition_error(err: Exception) -> bool:
    """True when yt-dlp failed to READ browser cookies (locked DB while the
    browser runs — yt-dlp #7271 — or a keyring/decryption hiccup), as opposed
    to the site rejecting the request. These are machine-local and the
    download itself hasn't been attempted yet, so retrying without cookies
    is safe and usually sufficient on a residential IP."""
    raw = str(err).lower()
    return "cookie" in raw and (
        "could not copy" in raw
        or "database is locked" in raw
        or "failed to decrypt" in raw
        or "could not find" in raw
    )


def _is_stale_partial_error(err: Exception) -> bool:
    """True when yt-dlp failed on ITS OWN local resume state — a `.ytdl`
    file referencing `.part-Frag*` pieces that no longer exist (the process
    was killed mid-download, or the retention sweep reclaimed some pieces).
    Surfaces as UnavailableVideoError wrapping FileNotFoundError. Retrying
    the resume fails identically; retrying from a clean slate succeeds —
    this is the signal that a clean retry is worth one shot.
    """
    raw = str(err)
    return "No such file or directory" in raw or "[Errno 2]" in raw


def _explain_download_failure(
    *,
    err: Exception,
    vod_url: str,
    platform: Platform,
    cookies_from_browser: str | None,
    cookies_file: str | None = None,
) -> IngestError:
    """Wrap yt-dlp's raw error with an actionable hint when we recognize the
    pattern. The two most common Windows foot-guns:
      * Kick 403 because no cookies were configured at all.
      * "Could not copy Chrome cookie database" because Chrome is running
        and holds an exclusive lock on its cookies SQLite file.
    """
    raw = str(err)
    # Chrome cookie-DB lock — yt-dlp issue #7271.
    if "Could not copy" in raw and "cookie" in raw.lower() and not cookies_file:
        return IngestError(
            f"yt-dlp can't read your browser's cookie database while the "
            f"browser is running (yt-dlp #7271). Three fixes, in order of "
            f"least friction:\n"
            f"  1. Switch to a browser that doesn't lock its cookie DB: "
            f"set CHALYBCLIP_COOKIES_FROM_BROWSER=edge (or firefox) in .env, "
            f"visit kick.com once in that browser, restart the server.\n"
            f"  2. Close Chrome completely (every chrome.exe process), then "
            f"retry. Chrome must stay closed during the run.\n"
            f"  3. Export cookies to a Netscape cookies.txt file (browser "
            f"extension 'Get cookies.txt LOCALLY'), set "
            f"CHALYBCLIP_COOKIES_FILE=C:/path/to/cookies.txt, restart the "
            f"server. Chrome can stay open.\n"
            f"Raw error: {raw}"
        )
    if (
        platform == "kick"
        and "403" in raw
        and not cookies_from_browser
        and not cookies_file
    ):
        return IngestError(
            f"yt-dlp 403'd on Kick for {vod_url}. Kick blocks unauthenticated "
            f"scraping; pass logged-in cookies through. On a server / container "
            f"deploy (no browser installed) export a Netscape cookies.txt while "
            f"logged into Kick, place it on the persistent volume, and set "
            f"CHALYBCLIP_COOKIES_FILE=/data/cookies.txt, then retry. For local dev "
            f"you can instead set CHALYBCLIP_COOKIES_FROM_BROWSER=chrome (or edge / "
            f"firefox / brave / chromium); that browser must have visited Kick at "
            f"least once. "
            f"Raw error: {raw}"
        )
    return IngestError(f"yt-dlp failed for {vod_url}: {raw}")


def _resolve_downloaded_path(info: dict[str, Any], *, fallback: Path) -> Path:
    """Best-effort lookup of the actual file yt-dlp wrote on disk."""
    requested = info.get("requested_downloads")
    if isinstance(requested, list) and requested:
        first = requested[0]
        if isinstance(first, dict):
            for key in ("filepath", "_filename"):
                value = first.get(key)
                if isinstance(value, str):
                    return Path(value)
    filename = info.get("_filename")
    if isinstance(filename, str):
        return Path(filename)
    return fallback


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract a 16 kHz mono PCM WAV from `video_path` using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as e:
        raise IngestError("ffmpeg binary not found on PATH") from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", "replace") if e.stderr else ""
        raise IngestError(f"ffmpeg audio extraction failed: {stderr}") from e


# A download is "complete enough" when the file on disk is within this
# fraction of the length the platform advertised. Below it, the fetch was
# truncated (still-processing upload, aborted segment stream) and we fail
# loudly rather than silently clipping the first few seconds.
_MIN_DOWNLOAD_COMPLETENESS = 0.9
# Absolute floor so a short clip legitimately off by a second or two
# (container rounding, trailing-ad trim) doesn't trip the ratio check.
_DOWNLOAD_SHORTFALL_TOLERANCE_S = 30.0


def _assert_download_complete(
    *, vod_url: str, claimed_s: float, actual_s: float
) -> None:
    """Guard against a truncated download.

    Compares the advertised duration (yt-dlp metadata) against what we
    actually got on disk (ffprobe). When the file is materially shorter
    than claimed, the source wasn't fully available — almost always a
    YouTube upload still being processed — so we raise instead of letting
    the pipeline run on a fragment and emit a degenerate clip.

    No-ops when either side is unknown (nothing claimed, or ffprobe
    unavailable) — there's no baseline to compare against.
    """
    if claimed_s <= 0.0 or actual_s <= 0.0:
        return
    if actual_s >= claimed_s * _MIN_DOWNLOAD_COMPLETENESS:
        return
    if claimed_s - actual_s <= _DOWNLOAD_SHORTFALL_TOLERANCE_S:
        return
    raise IngestError(
        f"download truncated: got {actual_s:.0f}s of a video the platform "
        f"reports as {claimed_s:.0f}s ({vod_url}). The video is likely still "
        "being processed after a recent upload — wait a few minutes and "
        "re-run. (Re-downloading replaces the partial file automatically.)"
    )


def _ffprobe_duration(video_path: Path) -> float:
    """Return the duration in seconds, or 0.0 if ffprobe fails."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError):
        return 0.0


def load_stream(stream_dir: Path) -> Stream:
    """Rehydrate a Stream from `<stream_dir>/stream.json`."""
    stream_json = Path(stream_dir) / "stream.json"
    if not stream_json.exists():
        raise IngestError(f"no stream at {stream_dir} (missing stream.json)")
    return Stream.model_validate_json(stream_json.read_text("utf-8"))
