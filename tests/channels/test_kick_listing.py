"""Kick channel-VOD listing via Kick's API.

yt-dlp has no Kick channel-VOD-list extractor and 403s behind Cloudflare,
so kick.com/<slug>/videos is listed through Kick's own API (curl_cffi
browser impersonation). These tests mock the HTTP fetch — no network.
"""

from __future__ import annotations

import pytest

from chalybclip.channels import service as svc


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://kick.com/n3on/videos", "n3on"),
        ("https://kick.com/n3on", "n3on"),
        ("kick.com/xqc/videos", "xqc"),
        ("https://www.kick.com/Some_Creator-1/videos", "Some_Creator-1"),
        ("https://kick.com/video/abc", None),  # reserved, not a channel
        ("https://youtube.com/@creator/videos", None),
    ],
)
def test_kick_channel_slug(url: str, expected: str | None) -> None:
    assert svc._kick_channel_slug(url) == expected


@pytest.mark.asyncio
async def test_list_kick_channel_vods_maps_api_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two entries, deliberately oldest-first to prove we sort newest-first.
    api_payload = [
        {
            "session_title": "Older stream",
            "created_at": "2026-06-14 10:00:00",
            "video": {"uuid": "11111111-1111-1111-1111-111111111111"},
        },
        {
            "session_title": "Newest stream",
            "created_at": "2026-06-15 22:00:00",
            "video": {"uuid": "22222222-2222-2222-2222-222222222222"},
        },
        # Malformed: no video.uuid — must be skipped, not crash.
        {"session_title": "Broken", "created_at": "2026-06-13 09:00:00"},
    ]
    monkeypatch.setattr(svc, "_fetch_kick_vods_sync", lambda slug: list(api_payload))

    vods = await svc.list_channel_vods(
        "kick", "https://kick.com/n3on/videos", limit=10
    )

    assert [v.video_id for v in vods] == [
        "22222222-2222-2222-2222-222222222222",  # newest first
        "11111111-1111-1111-1111-111111111111",
    ]
    newest = vods[0]
    assert newest.title == "Newest stream"
    # Ingestable URL matches yt-dlp's kick:vod extractor pattern.
    assert newest.url == (
        "https://kick.com/n3on/videos/22222222-2222-2222-2222-222222222222"
    )


@pytest.mark.asyncio
async def test_list_kick_channel_vods_respects_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [
        {
            "session_title": f"s{i}",
            "created_at": f"2026-06-{15 - i:02d} 00:00:00",
            "video": {"uuid": f"{i:08d}-0000-0000-0000-000000000000"},
        }
        for i in range(5)
    ]
    monkeypatch.setattr(svc, "_fetch_kick_vods_sync", lambda slug: list(payload))
    vods = await svc.list_channel_vods("kick", "https://kick.com/n3on", limit=2)
    assert len(vods) == 2


@pytest.mark.asyncio
async def test_list_kick_channel_vods_rejects_non_channel_url() -> None:
    with pytest.raises(ValueError, match="not a Kick channel URL"):
        await svc.list_channel_vods("kick", "https://kick.com/video/x", limit=5)
