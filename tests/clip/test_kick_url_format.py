"""Burned Kick-banner URL normalization (overlay_burn._format_kick_url).

The key guarantee: when no handle is set the export burns NOTHING (empty
string) — never a literal "KICK.COM/YOURHANDLE" placeholder on a published
clip. A real handle still normalizes to "KICK.COM/<HANDLE>".
"""

from __future__ import annotations

import pytest

from chalybclip.clip.overlay_burn import _format_kick_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("@", ""),
        ("kick.com/", ""),  # empty path → no placeholder
        ("aldo", "KICK.COM/ALDO"),
        ("@aldo", "KICK.COM/ALDO"),
        ("kick.com/aldo", "KICK.COM/ALDO"),
        ("https://kick.com/aldo", "KICK.COM/ALDO"),
        ("https://www.kick.com/aldo/", "KICK.COM/ALDO"),
    ],
)
def test_format_kick_url(raw: str, expected: str) -> None:
    assert _format_kick_url(raw) == expected


def test_empty_never_yields_placeholder() -> None:
    # Explicit guard against the regression: the old behavior burned
    # "KICK.COM/YOURHANDLE" into clips with no handle configured.
    assert "YOURHANDLE" not in _format_kick_url("")
