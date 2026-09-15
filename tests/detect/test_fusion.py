"""Tests for `detect_candidates` — voice + chat + audio + visual fusion."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from chalybclip.config import (
    AudioEnergyConfig,
    ChatHeatConfig,
    DetectionConfig,
    VisualConfig,
    VoiceDetectorConfig,
)
from chalybclip.detect import detect_candidates
from chalybclip.ingest import ChatMessage, ChatReplay, Stream
from chalybclip.vision import VisualSignal, VisualSignalTrack

from ._fixtures import make_stream, make_transcript


def _detection_config(
    *,
    voice_phrases: list[str] | None = None,
    chat_enabled: bool = True,
    audio_enabled: bool = False,
    visual_enabled: bool = False,
    merge_window_s: float = 30.0,
) -> DetectionConfig:
    return DetectionConfig(
        voice=VoiceDetectorConfig(
            enabled=True,
            weight=1.0,
            fuzzy_distance=2,
            phrases={"es": voice_phrases or ["clipéalo"]},
        ),
        chat_heat=ChatHeatConfig(
            enabled=chat_enabled,
            weight=0.7,
            baseline_window_s=10.0,
            spike_ratio=3.0,
            absolute_floor_msg_per_s=5.0,
        ),
        audio_energy=AudioEnergyConfig(
            enabled=audio_enabled,
            weight=0.5,
            frame_s=0.5,
            baseline_window_s=5.0,
            spike_ratio=2.5,
            sustain_s=1.5,
        ),
        visual=VisualConfig(
            enabled=visual_enabled,
            weight=0.6,
            cut_weight=1.0,
            emotion_weight=0.7,
            motion_weight=0.5,
            motion_baseline_window_s=10.0,
            motion_spike_ratio=2.0,
        ),
        merge_window_s=merge_window_s,
    )


def _write_loud_wav(path: Path, *, loud_start_s: float, loud_dur_s: float) -> None:
    rng = np.random.default_rng(0)
    sr = 16000
    parts: list[np.ndarray] = []
    for dur, amp in [
        (loud_start_s, 0.02),
        (loud_dur_s, 0.4),
        (5.0, 0.02),
    ]:
        n = int(sr * dur)
        if n <= 0:
            continue
        signal = rng.normal(0.0, amp, n)
        parts.append(np.clip(signal * 32767.0, -32768.0, 32767.0).astype(np.int16))
    full = np.concatenate(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(full.tobytes())


def _stream_with_audio(audio_path: Path) -> Stream:
    return Stream(
        id="str_01TEST",
        tenant_id="default",
        vod_url="https://kick.com/c/videos/1",
        platform="kick",
        title="t",
        channel="c",
        duration_s=600.0,
        source_video_path=audio_path.with_name("video.mp4"),
        source_audio_path=audio_path,
    )


def _chat_with_spike_at(ts: int, *, baseline_seconds: int = 10) -> ChatReplay:
    """Build chat replay where the `baseline_seconds` immediately preceding
    `ts` are quiet (1 msg/sec) and `ts` itself is a 20-msg spike."""
    msgs = [
        ChatMessage(ts=float(s), user="u", text="filler")
        for s in range(max(0, ts - baseline_seconds), ts)
    ]
    msgs.extend(
        ChatMessage(ts=float(ts) + i * 0.001, user="fan", text="hype")
        for i in range(20)
    )
    return ChatReplay(stream_id="str_01TEST", tenant_id="default", messages=msgs)


def test_voice_only_when_no_chat_replay() -> None:
    transcript = make_transcript(words=[(10.0, 11.0, " clipéalo", 0.9)])
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(),
        chat_replay=None,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "voice"


def test_chat_only_when_no_voice_match() -> None:
    transcript = make_transcript(words=[(0.5, 1.0, " hola", 0.9)])
    chat = _chat_with_spike_at(15)
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(),
        chat_replay=chat,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "chat"
    assert candidates[0].timestamp == 15.0


def test_voice_and_chat_fuse_within_merge_window() -> None:
    """A voice trigger at t=12 and a chat spike at t=15, both within
    merge_window_s=30. They should collapse to one cluster with both
    pieces of evidence under `evidence['matches']`."""
    transcript = make_transcript(words=[(12.0, 13.0, " clipéalo", 0.9)])
    chat = _chat_with_spike_at(15)
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(merge_window_s=30.0),
        chat_replay=chat,
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert "matches" in c.evidence
    assert len(c.evidence["matches"]) == 2
    reasons = {m.get("phrase") or m.get("ratio") for m in c.evidence["matches"]}
    # Either the phrase string or the ratio float — order-independent membership.
    assert any(r == "clipéalo" for r in reasons)


def test_voice_and_chat_far_apart_stay_separate() -> None:
    """Voice at t=12, chat spike at t=200 — well outside merge_window_s=30."""
    transcript = make_transcript(words=[(12.0, 13.0, " clipéalo", 0.9)])
    chat = _chat_with_spike_at(200)
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(merge_window_s=30.0),
        chat_replay=chat,
    )
    assert len(candidates) == 2
    reasons = sorted(c.reason for c in candidates)
    assert reasons == ["chat", "voice"]


def test_chat_disabled_still_returns_voice() -> None:
    """When chat_heat.enabled=False, fusion just returns voice candidates."""
    transcript = make_transcript(words=[(10.0, 11.0, " clipéalo", 0.9)])
    chat = _chat_with_spike_at(15)
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(chat_enabled=False),
        chat_replay=chat,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "voice"


def test_visual_only_fires_when_others_quiet() -> None:
    """Visual track with a smile transition produces a single visual candidate."""
    transcript = make_transcript(words=[(0.0, 0.5, " hola", 0.9)])
    track = VisualSignalTrack(
        stream_id="str_01TEST",
        tenant_id="default",
        signals=[
            VisualSignal(ts_offset_s=0.0, face_emotion="neutral"),
            VisualSignal(ts_offset_s=10.0, face_emotion="smile"),
        ],
    )
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(visual_enabled=True, chat_enabled=False),
        chat_replay=None,
        visual_track=track,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "visual"
    assert candidates[0].timestamp == 10.0


def test_voice_chat_audio_visual_all_four_fuse(tmp_path: Path) -> None:
    """All four detectors firing within merge_window_s collapse to one
    cluster carrying every signal under evidence['matches']."""
    audio_path = tmp_path / "audio.wav"
    _write_loud_wav(audio_path, loud_start_s=12.0, loud_dur_s=3.0)
    stream = _stream_with_audio(audio_path)

    transcript = make_transcript(words=[(12.0, 13.0, " clipéalo", 0.9)])
    chat = _chat_with_spike_at(15)
    visual_track = VisualSignalTrack(
        stream_id="str_01TEST",
        tenant_id="default",
        signals=[
            VisualSignal(ts_offset_s=float(s)) for s in range(11)
        ]
        + [VisualSignal(ts_offset_s=14.0, scene_cut=True, face_emotion="smile")],
    )

    candidates = detect_candidates(
        tenant_id="default",
        stream=stream,
        transcript=transcript,
        config=_detection_config(audio_enabled=True, visual_enabled=True),
        chat_replay=chat,
        visual_track=visual_track,
    )
    assert len(candidates) == 1
    matches = candidates[0].evidence["matches"]
    # Voice + chat + audio + visual all contributed.
    assert len(matches) >= 4


def test_visual_disabled_does_not_load_track() -> None:
    """Passing a track but config.visual.enabled=False is a no-op."""
    transcript = make_transcript(words=[(10.0, 11.0, " clipéalo", 0.9)])
    track = VisualSignalTrack(
        stream_id="str_01TEST",
        tenant_id="default",
        signals=[
            VisualSignal(ts_offset_s=0.0, face_emotion="neutral"),
            VisualSignal(ts_offset_s=5.0, face_emotion="smile"),
        ],
    )
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(visual_enabled=False),
        chat_replay=None,
        visual_track=track,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "voice"


def test_audio_voice_chat_all_three_fuse(tmp_path: Path) -> None:
    """Voice at 12s, audio spike around 13s, chat spike at 15s — all three
    within merge_window_s=30 should collapse into one cluster carrying
    all three pieces of evidence."""
    audio_path = tmp_path / "audio.wav"
    _write_loud_wav(audio_path, loud_start_s=12.0, loud_dur_s=3.0)
    stream = _stream_with_audio(audio_path)

    transcript = make_transcript(words=[(12.0, 13.0, " clipéalo", 0.9)])
    chat = _chat_with_spike_at(15)

    candidates = detect_candidates(
        tenant_id="default",
        stream=stream,
        transcript=transcript,
        config=_detection_config(audio_enabled=True),
        chat_replay=chat,
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert "matches" in c.evidence
    # Three signals firing in the same window all show up under matches.
    assert len(c.evidence["matches"]) >= 3


def test_audio_only_fires_when_chat_replay_absent(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    _write_loud_wav(audio_path, loud_start_s=10.0, loud_dur_s=3.0)
    stream = _stream_with_audio(audio_path)
    transcript = make_transcript(words=[(0.0, 0.5, " hola", 0.9)])
    candidates = detect_candidates(
        tenant_id="default",
        stream=stream,
        transcript=transcript,
        config=_detection_config(audio_enabled=True, chat_enabled=False),
        chat_replay=None,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "audio"


def test_audio_disabled_does_not_open_wav(tmp_path: Path) -> None:
    """Disabled detector returns empty without touching the audio file."""
    transcript = make_transcript(words=[(10.0, 11.0, " clipéalo", 0.9)])
    # Stream has no audio file on disk — detector should still skip cleanly.
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(audio_enabled=False, chat_enabled=False),
        chat_replay=None,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "voice"


def test_consecutive_chat_spike_seconds_collapse_to_one_candidate() -> None:
    """detect_chat_heat emits one candidate per spike second; fusion
    merges them via merge_window_s into a single cluster.

    The fourth spike second (sec 13) doesn't fire on its own — by then
    the prior spikes have lifted the rolling baseline above 5, so 15
    msg/sec is no longer 3x the baseline. The first three (secs 10, 11,
    12) all fire and get merged.
    """
    msgs = [ChatMessage(ts=float(s), user="u", text="f") for s in range(10)]
    for sec in (10, 11, 12, 13):
        msgs.extend(
            ChatMessage(ts=float(sec) + i * 0.001, user="fan", text="hype")
            for i in range(15)
        )
    chat = ChatReplay(stream_id="str_01TEST", tenant_id="default", messages=msgs)

    transcript = make_transcript(words=[(0.0, 0.5, " hola", 0.9)])
    candidates = detect_candidates(
        tenant_id="default",
        stream=make_stream(),
        transcript=transcript,
        config=_detection_config(merge_window_s=30.0),
        chat_replay=chat,
    )
    assert len(candidates) == 1
    assert candidates[0].reason == "chat"
    assert "matches" in candidates[0].evidence
    assert len(candidates[0].evidence["matches"]) == 3
