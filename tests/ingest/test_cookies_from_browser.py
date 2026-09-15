"""yt-dlp cookies-from-browser plumbing + Kick-403 friendlier error."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from chalybclip.errors import IngestError
from chalybclip.ingest.service import _download_vod, _explain_download_failure


class _FakeYoutubeDL:
    """Stand-in for yt_dlp.YoutubeDL — captures the opts the caller passed."""

    captured_opts: ClassVar[dict[str, Any]] = {}

    def __init__(self, opts: dict[str, Any]) -> None:
        type(self).captured_opts = opts

    def __enter__(self) -> _FakeYoutubeDL:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def extract_info(self, url: str, *, download: bool = True) -> dict[str, Any]:
        # Pretend yt-dlp downloaded a file at the requested outtmpl.
        outtmpl = type(self).captured_opts["outtmpl"]
        actual = Path(outtmpl.replace(".%(ext)s", ".mp4"))
        actual.parent.mkdir(parents=True, exist_ok=True)
        actual.write_bytes(b"\x00fakevideo")
        return {"duration": 60.0, "title": "x", "uploader": "y"}


# ---- cookies plumbing ----


def test_download_vod_passes_cookies_from_browser_when_set(tmp_path: Path) -> None:
    """When `cookies_from_browser="chrome"`, yt-dlp opts get the right tuple."""
    target = tmp_path / "vid" / "video.mp4"
    with patch("chalybclip.ingest.service.yt_dlp.YoutubeDL", _FakeYoutubeDL):
        _download_vod(
            vod_url="https://kick.com/x/videos/abc",
            target_path=target,
            cookies_from_browser="chrome",
            platform="kick",
        )
    assert _FakeYoutubeDL.captured_opts["cookiesfrombrowser"] == ("chrome",)


def test_download_vod_normalizes_browser_name(tmp_path: Path) -> None:
    """Mixed-case input is lowercased + stripped before hand-off."""
    target = tmp_path / "vid" / "video.mp4"
    with patch("chalybclip.ingest.service.yt_dlp.YoutubeDL", _FakeYoutubeDL):
        _download_vod(
            vod_url="https://kick.com/x/videos/abc",
            target_path=target,
            cookies_from_browser="  Chrome  ",
            platform="kick",
        )
    assert _FakeYoutubeDL.captured_opts["cookiesfrombrowser"] == ("chrome",)


def test_download_vod_omits_cookies_opt_when_none(tmp_path: Path) -> None:
    """No cookies setting -> no cookiesfrombrowser key in opts at all."""
    target = tmp_path / "vid" / "video.mp4"
    with patch("chalybclip.ingest.service.yt_dlp.YoutubeDL", _FakeYoutubeDL):
        _download_vod(
            vod_url="https://www.youtube.com/watch?v=abc",
            target_path=target,
            cookies_from_browser=None,
            platform="youtube",
        )
    assert "cookiesfrombrowser" not in _FakeYoutubeDL.captured_opts


def test_download_vod_omits_cookies_when_empty_string(tmp_path: Path) -> None:
    target = tmp_path / "vid" / "video.mp4"
    with patch("chalybclip.ingest.service.yt_dlp.YoutubeDL", _FakeYoutubeDL):
        _download_vod(
            vod_url="https://www.youtube.com/watch?v=abc",
            target_path=target,
            cookies_from_browser="",
            platform="youtube",
        )
    assert "cookiesfrombrowser" not in _FakeYoutubeDL.captured_opts


# ---- locked cookie DB self-heals: retry without cookies ----


def test_locked_cookie_db_retries_without_cookies(tmp_path: Path) -> None:
    """Chrome holding its cookie DB (yt-dlp #7271) must NOT fail the run:
    the download retries cookie-less instead of surfacing close-your-browser
    instructions — on a residential IP that retry almost always succeeds."""
    import yt_dlp

    attempts: list[dict[str, Any]] = []

    class _LockedThenOk(_FakeYoutubeDL):
        def extract_info(self, url: str, *, download: bool = True) -> dict[str, Any]:
            attempts.append(dict(type(self).captured_opts))
            if "cookiesfrombrowser" in type(self).captured_opts:
                raise yt_dlp.utils.DownloadError(
                    "ERROR: Could not copy Chrome cookie database"
                )
            return super().extract_info(url, download=download)

    target = tmp_path / "vid" / "video.mp4"
    with patch("chalybclip.ingest.service.yt_dlp.YoutubeDL", _LockedThenOk):
        _download_vod(
            vod_url="https://kick.com/x/videos/abc",
            target_path=target,
            cookies_from_browser="chrome",
            platform="kick",
        )
    assert len(attempts) == 2
    assert "cookiesfrombrowser" in attempts[0]
    assert "cookiesfrombrowser" not in attempts[1]
    assert target.exists()


def test_locked_cookie_db_then_site_error_still_explains(tmp_path: Path) -> None:
    """If the cookie-less retry ALSO fails (site genuinely wants cookies),
    the second error surfaces through the normal explain path."""
    import yt_dlp

    class _LockedThenForbidden(_FakeYoutubeDL):
        def extract_info(self, url: str, *, download: bool = True) -> dict[str, Any]:
            if "cookiesfrombrowser" in type(self).captured_opts:
                raise yt_dlp.utils.DownloadError(
                    "ERROR: Could not copy Chrome cookie database"
                )
            raise yt_dlp.utils.DownloadError(
                "ERROR: [kick:vod] abc: HTTP Error 403: Forbidden"
            )

    target = tmp_path / "vid" / "video.mp4"
    with patch("chalybclip.ingest.service.yt_dlp.YoutubeDL", _LockedThenForbidden):
        with pytest.raises(IngestError, match="403"):
            _download_vod(
                vod_url="https://kick.com/x/videos/abc",
                target_path=target,
                cookies_from_browser="chrome",
                platform="kick",
            )


# ---- friendlier error rewrite ----


def test_explain_kick_403_without_cookies_suggests_env_var() -> None:
    """The most common new-user foot-gun gets a specific, actionable message."""
    err = _explain_download_failure(
        err=Exception("ERROR: [kick:vod] abc: HTTP Error 403: Forbidden"),
        vod_url="https://kick.com/x/videos/abc",
        platform="kick",
        cookies_from_browser=None,
    )
    msg = str(err)
    assert isinstance(err, IngestError)
    assert "Kick blocks unauthenticated scraping" in msg
    assert "CHALYBCLIP_COOKIES_FROM_BROWSER" in msg
    assert "chrome" in msg


def test_explain_kick_403_with_cookies_set_falls_through_to_generic() -> None:
    """If cookies were already set and Kick still 403'd, the suggestion would
    be misleading. We pass through the raw error so the user can debug."""
    err = _explain_download_failure(
        err=Exception("ERROR: [kick:vod] abc: HTTP Error 403: Forbidden"),
        vod_url="https://kick.com/x/videos/abc",
        platform="kick",
        cookies_from_browser="chrome",
    )
    assert "CHALYBCLIP_COOKIES_FROM_BROWSER" not in str(err)
    assert "yt-dlp failed" in str(err)


def test_explain_non_kick_403_falls_through_to_generic() -> None:
    """A 403 from YouTube isn't the same problem; don't suggest Kick remedy."""
    err = _explain_download_failure(
        err=Exception("ERROR: [youtube] abc: HTTP Error 403: Forbidden"),
        vod_url="https://www.youtube.com/watch?v=abc",
        platform="youtube",
        cookies_from_browser=None,
    )
    assert "Kick" not in str(err)
    assert "yt-dlp failed" in str(err)


def test_explain_kick_non_403_error_falls_through(tmp_path: Path) -> None:
    """Non-403 Kick errors (e.g. network) are not the cookie-missing pattern."""
    err = _explain_download_failure(
        err=Exception("ERROR: [kick:vod] abc: timed out"),
        vod_url="https://kick.com/x/videos/abc",
        platform="kick",
        cookies_from_browser=None,
    )
    assert "CHALYBCLIP_COOKIES_FROM_BROWSER" not in str(err)


# ---- ingest_vod wires Settings through ----


@pytest.mark.asyncio
async def test_ingest_vod_picks_cookies_from_settings_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the kwarg isn't passed, ingest_vod reads CHALYBCLIP_COOKIES_FROM_BROWSER."""
    from chalybclip.ingest.service import ingest_vod
    from chalybclip.settings import get_settings

    monkeypatch.setenv("CHALYBCLIP_COOKIES_FROM_BROWSER", "chrome")
    get_settings.cache_clear()

    captured: list[str | None] = []

    def fake_download(
        *,
        vod_url: str,
        target_path: Path,
        cookies_from_browser: str | None = None,
        cookies_file: str | None = None,
        platform: Any = "unknown",
    ) -> dict[str, Any]:
        captured.append(cookies_from_browser)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"\x00fakevideo")
        return {"duration": 60.0, "title": "x", "uploader": "y"}

    def fake_extract_audio(_video: Path, audio_path: Path) -> None:
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"\x00fakeaudio")

    monkeypatch.setattr("chalybclip.ingest.service._download_vod", fake_download)
    monkeypatch.setattr("chalybclip.ingest.service._extract_audio", fake_extract_audio)

    try:
        await ingest_vod(
            tenant_id="default",
            vod_url="https://kick.com/x/videos/abc",
            output_dir=tmp_path,
        )
    finally:
        get_settings.cache_clear()  # don't pollute other tests
    assert captured == ["chrome"]
