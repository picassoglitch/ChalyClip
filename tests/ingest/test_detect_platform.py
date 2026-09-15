"""Pure unit tests for platform detection."""

from __future__ import annotations

import pytest

from chalybclip.ingest.service import detect_platform


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://kick.com/aldovillanueva/videos/abc123", "kick"),
        ("https://www.kick.com/aldovillanueva/videos/abc123", "kick"),
        ("https://KICK.COM/aldo/videos/x", "kick"),
        ("https://www.twitch.tv/videos/123456789", "twitch"),
        ("https://twitch.tv/videos/123456789", "twitch"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
        ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
        ("https://example.com/some/video", "unknown"),
        ("https://kick.example.com/fake", "unknown"),
        ("https://twitch.tv.malicious.com/x", "unknown"),
    ],
)
def test_detect_platform(url: str, expected: str) -> None:
    assert detect_platform(url) == expected
