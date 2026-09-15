"""Chat replay persistence — JSONL on disk, one message per line.

Phase 1 ships the load/save helpers and the chat heat detector that
consumes them. Live fetchers for Kick / Twitch / YouTube are deferred
to Phase 2 when we can verify each platform's API shape against real
streams; for Phase 1 the user (or a future fetcher) hands us a
pre-fetched JSONL via `--chat-replay <file>` on the CLI.

The JSONL is the canonical on-disk shape for a stream's chat:

    <stream_dir>/source/chat.jsonl

Each line is a `ChatMessage` JSON object. Sorted by `ts` ascending so
the chat heat detector can sweep linearly.
"""

from __future__ import annotations

from pathlib import Path

from chalybclip.errors import IngestError

from .models import ChatMessage, ChatReplay


def chat_replay_path(stream_dir: Path) -> Path:
    """Canonical chat replay file for a stream."""
    return Path(stream_dir) / "source" / "chat.jsonl"


def save_chat_replay(stream_dir: Path, replay: ChatReplay) -> Path:
    """Write `replay` to `<stream_dir>/source/chat.jsonl`, sorted by ts."""
    path = chat_replay_path(stream_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_msgs = sorted(replay.messages, key=lambda m: m.ts)
    with path.open("w", encoding="utf-8") as f:
        for msg in sorted_msgs:
            f.write(msg.model_dump_json() + "\n")
    return path


def load_chat_replay(stream_dir: Path, *, stream_id: str, tenant_id: str) -> ChatReplay | None:
    """Read `chat.jsonl` if it exists, else return None.

    The detector treats `None` as "no chat signal" — silently skipped,
    not an error. Platforms without chat replay (or VODs that just
    don't have chat) get this branch.
    """
    path = chat_replay_path(stream_dir)
    if not path.exists():
        return None
    messages: list[ChatMessage] = []
    for line in path.read_text("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        messages.append(ChatMessage.model_validate_json(stripped))
    messages.sort(key=lambda m: m.ts)
    return ChatReplay(stream_id=stream_id, tenant_id=tenant_id, messages=messages)


def import_chat_replay(
    *,
    source: Path,
    stream_dir: Path,
    stream_id: str,
    tenant_id: str,
) -> ChatReplay:
    """Read a JSONL from `source`, normalize, write into the stream's chat.jsonl."""
    src = Path(source)
    if not src.exists():
        raise IngestError(f"chat replay source not found: {src}")
    messages: list[ChatMessage] = []
    for line_num, line in enumerate(src.read_text("utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            messages.append(ChatMessage.model_validate_json(stripped))
        except Exception as e:
            raise IngestError(
                f"chat replay line {line_num} did not validate against ChatMessage: {e}"
            ) from e
    replay = ChatReplay(stream_id=stream_id, tenant_id=tenant_id, messages=messages)
    save_chat_replay(stream_dir, replay)
    return replay
