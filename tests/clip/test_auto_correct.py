"""auto_correct_clip — slice G.6.

The pipeline calls this per clip right after the cut, while the source VOD
still exists. It must: auto-trim around integrity issues, apply the safe
overlay fixes, record a human-readable summary to the sidecar + event log,
and report the before/after publishability. We fake the DB-backed pieces
(breakdown, trim, repos) but use the REAL apply_ai_fixes /
compute_publishability so the report reflects production scoring.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from chalybclip.clip import auto_correct as ac
from chalybclip.clip.auto_correct import auto_correct_clip
from chalybclip.clip.breakdown import ClipBreakdown


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Insulate from the suite-wide structlog leak (see test_trim_fallback)."""
    import structlog

    structlog.reset_defaults()


@dataclass
class _FakeClip:
    id: str
    stream_id: str
    path: Path
    start_s: float
    end_s: float
    duration_s: float
    status: str = "cut"
    overlay_config: dict | None = None
    original_start_s: float | None = None
    original_end_s: float | None = None
    publishability_score: int | None = None
    publishability_status: str | None = None


@dataclass
class _FakeClipsRepo:
    def __init__(self, _db: object) -> None:
        self.clip = _STATE["clip"]
        self.calls = _STATE["calls"]

    async def get(self, clip_id: str):
        return self.clip if clip_id == self.clip.id else None

    async def set_overlay_config(self, clip_id: str, *, overlay_config: dict) -> None:
        self.clip.overlay_config = overlay_config
        self.calls.append(("set_overlay_config", overlay_config))

    async def set_publishability(self, clip_id: str, *, score: int, status: str) -> None:
        self.clip.publishability_score = score
        self.clip.publishability_status = status
        self.calls.append(("set_publishability", score, status))

    async def update_status(self, clip_id: str, *, status: str):
        self.clip.status = status
        self.calls.append(("update_status", status))
        return self.clip


@dataclass
class _FakeEventsRepo:
    def __init__(self, _db: object) -> None:
        self.events = _STATE["events"]

    async def emit(self, *, type: str, payload: dict) -> None:
        self.events.append((type, payload))


_STATE: dict = {}


def _breakdown(clip_id: str, duration_s: float, issues: tuple) -> ClipBreakdown:
    return ClipBreakdown(
        clip_id=clip_id,
        duration_s=duration_s,
        motion_score=0.5,
        face_presence=0.8,
        speaking_intensity=2.5,
        reaction_confidence=0.7,
        rescore_delta=0.1,
        rescore_reason=None,
        heuristic_reason="voice",
        heuristic_score=0.7,
        integrity_issues=issues,
    )


def _wire(monkeypatch: pytest.MonkeyPatch, *, clip: _FakeClip, breakdowns: list) -> dict:
    """breakdowns: list returned in sequence by the faked clip_breakdown."""
    _STATE.clear()
    _STATE.update(clip=clip, calls=[], events=[])

    seq = list(breakdowns)

    async def fake_breakdown(_db, _clip_id):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    async def fake_trim(_db, *, clip_id, integrity_issues, **_kw):
        # Simulate a successful trim: shorten the clip + stash originals.
        clip.original_start_s = clip.start_s
        clip.original_end_s = clip.end_s
        clip.start_s = 0.0
        clip.end_s = 15.0
        clip.duration_s = 15.0
        return {
            "outcome": "trimmed",
            "new_start_s": 0.0,
            "new_end_s": 15.0,
            "new_duration_s": 15.0,
        }

    monkeypatch.setattr(ac, "clip_breakdown", fake_breakdown)
    monkeypatch.setattr(ac, "auto_trim_around_integrity", fake_trim)
    monkeypatch.setattr(ac, "ClipsRepo", _FakeClipsRepo)
    monkeypatch.setattr(ac, "EventsRepo", _FakeEventsRepo)
    return _STATE


def test_auto_correct_trims_fixes_and_records_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_dir = tmp_path / "clp_X"
    clip_dir.mkdir()
    clip_mp4 = clip_dir / "clip.mp4"
    clip_mp4.write_bytes(b"clip")

    clip = _FakeClip(
        id="clp_X", stream_id="str_X", path=clip_mp4,
        start_s=0.0, end_s=40.0, duration_s=40.0,
        # captions disabled → apply_ai_fixes will flip them on (one fix).
        overlay_config={"captions": {"enabled": False}},
    )
    issues = (
        {"kind": "freeze", "start_s": 15.0, "end_s": 24.0, "label": "freeze"},
        {"kind": "freeze", "start_s": 25.0, "end_s": 40.0, "label": "freeze"},
    )
    # First breakdown (pre-trim) has the issues; second (post-trim) is clean.
    state = _wire(
        monkeypatch,
        clip=clip,
        breakdowns=[
            _breakdown("clp_X", 40.0, issues),
            _breakdown("clp_X", 15.0, ()),
        ],
    )

    report = asyncio.run(auto_correct_clip(object(), clip_id="clp_X"))

    kinds = [c["kind"] for c in report["corrections"]]
    assert "trim" in kinds            # auto-trim ran
    assert "overlay" in kinds         # captions were enabled
    assert report["trimmed"] is True
    assert report["score_after"] >= report["score_before"]

    # The trim label names what was removed, in Spanish.
    trim_label = next(c["label"] for c in report["corrections"] if c["kind"] == "trim")
    assert "imagen congelada" in trim_label

    # Overlay was persisted with captions enabled.
    assert clip.overlay_config["captions"]["enabled"] is True

    # Sidecar written next to the clip MP4, matching the report.
    sidecar = clip_dir / ac.AUTO_CORRECTIONS_FILENAME
    assert sidecar.exists()
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    assert saved["corrections"] == report["corrections"]

    # Event emitted for observability.
    assert any(t == "clip.auto_corrected" for t, _ in state["events"])


def test_auto_correct_noop_when_clean_and_already_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_dir = tmp_path / "clp_Y"
    clip_dir.mkdir()
    clip_mp4 = clip_dir / "clip.mp4"
    clip_mp4.write_bytes(b"clip")

    clip = _FakeClip(
        id="clp_Y", stream_id="str_Y", path=clip_mp4,
        start_s=0.0, end_s=30.0, duration_s=30.0,
        overlay_config={"captions": {"enabled": True}, "safe_zone_platform": "tiktok"},
    )
    state = _wire(
        monkeypatch, clip=clip, breakdowns=[_breakdown("clp_Y", 30.0, ())]
    )

    report = asyncio.run(auto_correct_clip(object(), clip_id="clp_Y"))

    assert report["trimmed"] is False
    assert report["corrections"] == []
    # No overlay write when nothing changed.
    assert not any(c[0] == "set_overlay_config" for c in state["calls"])
    # Sidecar is still written (records the clean verdict).
    assert (clip_dir / ac.AUTO_CORRECTIONS_FILENAME).exists()
