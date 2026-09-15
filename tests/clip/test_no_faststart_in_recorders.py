"""Render Migration R8 — pin that the render-path ffmpeg cmds DO NOT
pass `+faststart`.

Root cause of two prod failure sessions: ffmpeg's two-pass faststart
re-opens the output file after the first write to relocate the moov
atom. On Railway's persistent volume that re-open fails ("Unable to
re-open output file for shifting data"), and ffmpeg cleans up the
file as it bails. Result: rc=0 with no output → R5 raises loudly →
operator sees a failed render with no way to know why for a long
time.

Faststart is purely an optimization for in-browser progressive
streaming of an MP4 hosted at a URL. The operator's flow is download
+ local playback + platform upload; none of those care about moov
position. If we ever need streaming preview, run a SEPARATE
qt-faststart pass that doesn't have ffmpeg's delete-on-failure
semantics.

Test is AST-level: grep the file for `+faststart` token. Belt-and-
braces against a future "well, this one ffmpeg call is fine, just
this once" addition that re-introduces the bug.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2] / "chalybclip"


# Files that produce the FINAL rendered MP4 (the one downloaded /
# uploaded). These must not use +faststart.
_RENDER_PATHS = [
    _ROOT / "clip" / "hybrid_recorder.py",
    _ROOT / "clip" / "preview_recorder.py",
]


@pytest.mark.parametrize("path", _RENDER_PATHS)
def test_render_path_does_not_use_faststart(path: Path) -> None:
    """Each render-stage ffmpeg cmd must NOT pass +faststart.

    If a future commit re-adds it, this test fails immediately with a
    pointer to the rationale rather than waiting for a Railway
    production failure to expose the same bug again.
    """
    text = path.read_text(encoding="utf-8")
    # Be specific: we want to catch the literal arg passed to ffmpeg,
    # not casual mentions in comments/docstrings explaining why we
    # don't use it. The marker for "this is the arg" is the leading
    # quoted "+faststart" passed as a list item, almost always
    # preceded by `"-movflags",` on the line before.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip comments / docstrings.
        if stripped.startswith("#"):
            continue
        if '"+faststart"' in line and '"-movflags"' in (
            lines[i - 1] if i > 0 else ""
        ):
            pytest.fail(
                f"{path.name}:{i + 1} re-introduces +faststart. "
                "Drop it — see R8 commit; ffmpeg's faststart re-open "
                "fails on Railway and deletes the output."
            )
