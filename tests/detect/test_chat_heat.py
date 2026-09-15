"""Tests for the chat heat detector."""

from __future__ import annotations

import pytest

from chalybclip.config import ChatHeatConfig
from chalybclip.detect import detect_chat_heat
from chalybclip.errors import DetectionError
from chalybclip.ingest import ChatMessage, ChatReplay

from ._fixtures import make_stream


def _replay(*pairs: tuple[float, str, str], tenant_id: str = "default", stream_id: str = "str_01TEST") -> ChatReplay:
    return ChatReplay(
        stream_id=stream_id,
        tenant_id=tenant_id,
        messages=[ChatMessage(ts=ts, user=u, text=t) for ts, u, t in pairs],
    )


def _enabled_config(
    *,
    weight: float = 0.7,
    baseline_window_s: float = 10.0,
    spike_ratio: float = 3.0,
    absolute_floor_msg_per_s: float = 5.0,
) -> ChatHeatConfig:
    return ChatHeatConfig(
        enabled=True,
        weight=weight,
        baseline_window_s=baseline_window_s,
        spike_ratio=spike_ratio,
        absolute_floor_msg_per_s=absolute_floor_msg_per_s,
    )


def test_disabled_detector_returns_empty() -> None:
    cfg = ChatHeatConfig(enabled=False)
    replay = _replay((0.0, "u", "x"))
    assert detect_chat_heat("default", make_stream(), replay, cfg) == []


def test_empty_replay_returns_empty() -> None:
    assert detect_chat_heat("default", make_stream(), _replay(), _enabled_config()) == []


def test_single_spike_detected() -> None:
    """Quiet baseline (1 msg/sec for 10 s) followed by 20 messages in one
    second — well over the spike_ratio threshold."""
    msgs: list[tuple[float, str, str]] = []
    # 1 message per second for the first 10 seconds.
    for sec in range(10):
        msgs.append((float(sec), "u", "filler"))
    # 20 messages at second 10 — the spike.
    for i in range(20):
        msgs.append((10.0 + i * 0.001, f"u{i}", f"hype {i}"))
    replay = _replay(*msgs)
    cands = detect_chat_heat("default", make_stream(), replay, _enabled_config())
    assert len(cands) == 1
    c = cands[0]
    assert c.timestamp == 10.0
    assert c.reason == "chat"
    assert c.evidence["spike_rate_per_s"] == 20
    assert c.evidence["baseline_rate_per_s"] == pytest.approx(1.0)
    assert c.evidence["ratio"] == pytest.approx(20.0)
    # Score is weight * min(1, ratio / spike_ratio); 20/3 > 1, so score = weight.
    assert c.score == pytest.approx(0.7)
    assert len(c.evidence["sample_messages"]) == 3


def test_spike_below_absolute_floor_ignored() -> None:
    """Even a high-ratio spike skips when current rate is below the floor."""
    # 0.1 msg/sec baseline; 3 messages in one second is 30x ratio but
    # below absolute_floor_msg_per_s = 5.
    msgs: list[tuple[float, str, str]] = [(float(i), "u", "x") for i in range(10) if i % 5 == 0]
    msgs.extend([(15.0, "u", f"m{i}") for i in range(3)])
    replay = _replay(*msgs)
    cfg = _enabled_config(absolute_floor_msg_per_s=5.0)
    assert detect_chat_heat("default", make_stream(), replay, cfg) == []


def test_high_baseline_no_ratio_spike_ignored() -> None:
    """Once the rolling baseline saturates at 10 msg/sec, a second with 12
    messages is above the absolute floor but only 1.2x baseline -- under
    spike_ratio = 3.0. We assert that the saturated-baseline second
    (sec 30) does NOT fire. (Earlier seconds may fire while the baseline
    ramps up -- that's correct behavior, not what we're testing here.)
    """
    msgs: list[tuple[float, str, str]] = []
    # 30 seconds of 10 msg/sec so the rolling baseline has fully saturated
    # before the test second.
    for sec in range(30):
        for j in range(10):
            msgs.append((sec + j * 0.01, "u", "hi"))
    # 12 messages at sec 30 -- only 1.2x the saturated baseline.
    for j in range(12):
        msgs.append((30.0 + j * 0.001, "u", "still chatting"))
    replay = _replay(*msgs)
    cands = detect_chat_heat("default", make_stream(), replay, _enabled_config())
    spike_secs = {int(c.timestamp) for c in cands}
    assert 30 not in spike_secs


def test_score_scales_with_ratio_up_to_full_weight() -> None:
    """Score = weight * min(1, ratio / spike_ratio). A 6x spike (2x the
    threshold) should saturate at weight (no double counting)."""
    msgs: list[tuple[float, str, str]] = [(float(s), "u", "f") for s in range(10)]
    for j in range(6):
        msgs.append((10.0 + j * 0.001, "u", f"big {j}"))
    replay = _replay(*msgs)
    cfg = _enabled_config(weight=0.5, absolute_floor_msg_per_s=3.0)
    cands = detect_chat_heat("default", make_stream(), replay, cfg)
    assert len(cands) == 1
    assert cands[0].score == pytest.approx(0.5)


def test_tenant_mismatch_raises() -> None:
    replay = _replay((0.0, "u", "x"), tenant_id="ten_b")
    with pytest.raises(DetectionError, match="tenant mismatch"):
        detect_chat_heat("ten_a", make_stream(tenant_id="ten_a"), replay, _enabled_config())


def test_stream_mismatch_raises() -> None:
    replay = _replay((0.0, "u", "x"), stream_id="str_OTHER")
    with pytest.raises(DetectionError, match="stream/chat mismatch"):
        detect_chat_heat("default", make_stream(stream_id="str_01TEST"), replay, _enabled_config())


def test_consecutive_spike_seconds_each_emit_candidate() -> None:
    """A spike that lasts 3 seconds emits 3 raw candidates; the fusion
    step (detect_candidates) merges them later via merge_window_s."""
    msgs: list[tuple[float, str, str]] = [(float(s), "u", "f") for s in range(10)]
    for sec in (10, 11, 12):
        for j in range(15):
            msgs.append((sec + j * 0.001, "u", "hype"))
    replay = _replay(*msgs)
    cands = detect_chat_heat("default", make_stream(), replay, _enabled_config())
    assert [int(c.timestamp) for c in cands] == [10, 11, 12]
