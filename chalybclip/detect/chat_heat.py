"""Chat heat detector.

Algorithm (rolling baseline + spike test):
    1. Bucket chat messages into 1-second bins.
    2. For each second t with messages, compute the baseline as the mean
       msg/sec over [t - baseline_window_s, t).
    3. A spike is `current_rate > spike_ratio * baseline` AND
       `current_rate >= absolute_floor_msg_per_s` (the absolute floor
       suppresses noise from sparse-chat streams).
    4. Each spike second emits one Candidate with `reason="chat"` and the
       sample messages from that second as evidence. Adjacent spikes get
       merged later by the same `_merge_candidates` step that handles
       voice clusters.

Linear time over the number of seconds with at least one chat message
via a sliding-window deque.
"""

from __future__ import annotations

from collections import defaultdict, deque

from chalybclip.config import ChatHeatConfig
from chalybclip.errors import DetectionError
from chalybclip.ingest import ChatMessage, ChatReplay, Stream

from .models import Candidate

_SAMPLE_MESSAGES_PER_SPIKE = 3


def detect_chat_heat(
    tenant_id: str,
    stream: Stream,
    chat_replay: ChatReplay,
    config: ChatHeatConfig,
) -> list[Candidate]:
    """Return chat-heat candidates for `stream`.

    Returns `[]` when the detector is disabled or there's no chat replay.
    Tenant-mismatch raises (CLAUDE.md hard rule #1).
    """
    if not config.enabled or not chat_replay.messages:
        return []
    if tenant_id != stream.tenant_id:
        raise DetectionError(f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}")
    if tenant_id != chat_replay.tenant_id:
        raise DetectionError(
            f"tenant mismatch: caller={tenant_id!r}, chat={chat_replay.tenant_id!r}"
        )
    if stream.id != chat_replay.stream_id:
        raise DetectionError(
            f"stream/chat mismatch: stream={stream.id} chat={chat_replay.stream_id}"
        )

    # Bin messages by integer second.
    rates: dict[int, int] = defaultdict(int)
    msgs_at: dict[int, list[ChatMessage]] = defaultdict(list)
    for msg in chat_replay.messages:
        sec = int(msg.ts)
        rates[sec] += 1
        msgs_at[sec].append(msg)

    sorted_secs = sorted(rates)
    if not sorted_secs:
        return []

    # Sliding-window mean rate over [t - baseline_window_s, t).
    window: deque[tuple[int, int]] = deque()
    window_sum = 0
    candidates: list[Candidate] = []

    for t in sorted_secs:
        # Window is [t - baseline_window_s, t) — inclusive lower bound, so
        # the second exactly `baseline_window_s` ago is still in-window.
        cutoff = t - config.baseline_window_s
        while window and window[0][0] < cutoff:
            _, dropped = window.popleft()
            window_sum -= dropped

        # baseline = total messages in window / window length in seconds.
        baseline = window_sum / config.baseline_window_s

        current = rates[t]
        if (
            current >= config.absolute_floor_msg_per_s
            and baseline > 0
            and current > config.spike_ratio * baseline
        ):
            ratio = current / baseline
            score = config.weight * min(1.0, ratio / config.spike_ratio)
            sample = msgs_at[t][:_SAMPLE_MESSAGES_PER_SPIKE]
            candidates.append(
                Candidate(
                    timestamp=float(t),
                    score=score,
                    reason="chat",
                    evidence={
                        "spike_rate_per_s": current,
                        "baseline_rate_per_s": baseline,
                        "ratio": ratio,
                        "sample_messages": [{"user": m.user, "text": m.text} for m in sample],
                    },
                )
            )

        # Add this second so future iterations see it as history.
        window.append((t, rates[t]))
        window_sum += rates[t]

    return candidates
