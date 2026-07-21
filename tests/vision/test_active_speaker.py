"""Tests for the heuristic active-speaker scorer + its pure helpers.

All pure Python — no OpenCV, no ffmpeg. The lip-motion / audio-sync
decision maths, the mouth-ROI geometry, and the stdlib audio-envelope
reduction are exercised directly with synthetic inputs, so a wrong
correlation or a mis-clamped ROI is a test failure rather than something
only a real video would surface.
"""

from __future__ import annotations

import struct

import pytest

from nexoclip.vision.active_speaker import (
    motion_magnitude,
    mouth_roi_box,
    pick_active_speaker,
    score_candidate,
)
from nexoclip.vision.face_track import _rms_bins, _sample_envelope

# ---- pick_active_speaker: contract edges ----


def test_empty_returns_none() -> None:
    assert pick_active_speaker([]) is None


def test_single_candidate_returns_zero() -> None:
    """One face -> nothing to disambiguate, always index 0 (even if still)."""
    assert pick_active_speaker([[0.0, 0.0, 0.0]]) == 0
    assert pick_active_speaker([[9.0, 8.0, 9.0]]) == 0


# ---- pick_active_speaker: lip motion ----


def test_moving_mouth_beats_still_mouth() -> None:
    moving = [10.0, 12.0, 9.0, 11.0]
    still = [0.0, 1.0, 0.0, 0.0]
    assert pick_active_speaker([moving, still]) == 0
    # Order must not bias the result — the moving mouth wins either slot.
    assert pick_active_speaker([still, moving]) == 1


def test_all_below_min_motion_abstains() -> None:
    """When no face's mouth is visibly moving, the scorer returns None so
    the caller falls back to its size/persistence heuristic."""
    quiet = [[0.0, 0.2, 0.1], [0.3, 0.0, 0.2]]
    assert pick_active_speaker(quiet, min_motion=1.5) is None


def test_default_min_motion_always_decides() -> None:
    """With the default gate (0.0) even tiny motion yields a decision."""
    assert pick_active_speaker([[0.0, 0.1], [0.4, 0.5]]) == 1


# ---- pick_active_speaker: audio-visual sync ----


def test_audio_energy_breaks_a_tie() -> None:
    """Two mouths with identical motion MAGNITUDE but opposite phase: with
    no audio the tie resolves to the lowest index; a correlated audio
    envelope flips the winner to the mouth moving in time with the sound."""
    cand0 = [2.0, 0.0, 2.0, 0.0]
    cand1 = [0.0, 2.0, 0.0, 2.0]
    assert motion_magnitude(cand0) == motion_magnitude(cand1)  # genuine tie

    assert pick_active_speaker([cand0, cand1]) == 0  # tie -> lowest index
    audio = [0.0, 2.0, 0.0, 2.0]  # syncs with cand1
    assert pick_active_speaker([cand0, cand1], audio_energy=audio) == 1


def test_audio_cannot_veto_a_clearly_stronger_mouth() -> None:
    """Sync is additive and non-negative — it breaks ties, it does not pull
    a much-louder mouth below a quiet-but-synced one."""
    loud = [10.0, 10.0, 10.0, 10.0]  # big magnitude, flat (no phase)
    quiet_synced = [0.0, 2.0, 0.0, 2.0]  # small magnitude, perfect sync
    audio = [0.0, 2.0, 0.0, 2.0]
    assert pick_active_speaker([loud, quiet_synced], audio_energy=audio) == 0


def test_mismatched_audio_length_is_ignored_safely() -> None:
    """A shorter/mismatched audio window contributes no sync term (rather
    than crashing), so the decision falls back to motion magnitude."""
    strong = [5.0, 6.0, 5.0, 6.0]
    weak = [1.0, 0.0, 1.0, 0.0]
    assert pick_active_speaker([strong, weak], audio_energy=[1.0, 2.0]) == 0


# ---- helpers: motion_magnitude / score_candidate ----


def test_motion_magnitude_is_mean_and_empty_safe() -> None:
    assert motion_magnitude([2.0, 4.0, 6.0]) == 4.0
    assert motion_magnitude([]) == 0.0


def test_score_candidate_adds_positive_sync_only() -> None:
    motion = [0.0, 2.0, 0.0, 2.0]
    scale = 1.0
    base = score_candidate(motion, None, audio_weight=0.5, motion_scale=scale)
    synced = score_candidate(
        motion, [0.0, 2.0, 0.0, 2.0], audio_weight=0.5, motion_scale=scale
    )
    anti = score_candidate(
        motion, [2.0, 0.0, 2.0, 0.0], audio_weight=0.5, motion_scale=scale
    )
    assert synced > base  # correlated audio raises the score
    assert anti == base  # anti-correlated audio is clamped to +0, not negative


# ---- mouth_roi_box: geometry ----


def test_mouth_roi_is_lower_third_and_inset() -> None:
    # 60x60 face at (100, 50); inset = 60//6 = 10; lower third starts at
    # y + 2h/3 = 50 + 40 = 90.
    box = mouth_roi_box(100, 50, 60, 60, frame_w=480, frame_h=270)
    assert box == (110, 90, 150, 110)


def test_mouth_roi_clamps_to_frame_bounds() -> None:
    # Face hard against the right/bottom edge — the ROI never exceeds frame.
    x0, y0, x1, y1 = mouth_roi_box(450, 240, 60, 60, frame_w=480, frame_h=270)
    assert 0 <= x0 <= x1 <= 480
    assert 0 <= y0 <= y1 <= 270


def test_mouth_roi_degenerate_rect_is_zero_area() -> None:
    x0, _y0, x1, _y1 = mouth_roi_box(10, 10, 0, 0, frame_w=480, frame_h=270)
    assert x1 == x0  # zero-width -> caller reads "no motion sample"


# ---- audio envelope reduction (stdlib, face_track) ----


def test_rms_bins_reduces_pcm_to_normalized_energy() -> None:
    # 8 kHz, 25 bins/s -> 320 samples/bin. One half-scale bin, one silent.
    samples = [16384] * 320 + [0] * 320
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    bins = _rms_bins(pcm, sr=8000, bins_per_s=25)
    assert bins is not None
    assert len(bins) == 2
    assert bins[0] == pytest.approx(0.5, abs=1e-3)  # 16384 / 32768
    assert bins[1] == pytest.approx(0.0, abs=1e-6)


def test_rms_bins_empty_pcm_returns_none() -> None:
    assert _rms_bins(b"", sr=8000, bins_per_s=25) is None


def test_sample_envelope_indexes_by_clip_time() -> None:
    env = [0.1, 0.2, 0.3, 0.4]
    assert _sample_envelope(env, 0.0, 4.0) == 0.1
    assert _sample_envelope(env, 2.0, 4.0) == 0.3
    assert _sample_envelope(env, 3.9, 4.0) == 0.4
    # Clamp past the end and on empty input.
    assert _sample_envelope(env, 99.0, 4.0) == 0.4
    assert _sample_envelope([], 1.0, 4.0) == 0.0
