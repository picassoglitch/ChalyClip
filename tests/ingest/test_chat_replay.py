"""Tests for chat-replay JSONL persistence + import."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.errors import IngestError
from chalybclip.ingest import (
    ChatMessage,
    ChatReplay,
    chat_replay_path,
    import_chat_replay,
    load_chat_replay,
    save_chat_replay,
)


def _replay(*pairs: tuple[float, str, str]) -> ChatReplay:
    return ChatReplay(
        stream_id="str_01ABC",
        tenant_id="ten_a",
        messages=[ChatMessage(ts=ts, user=u, text=t) for ts, u, t in pairs],
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    stream_dir = tmp_path / "str_01ABC"
    replay = _replay(
        (10.0, "alice", "first"),
        (12.5, "bob", "second"),
    )
    save_chat_replay(stream_dir, replay)
    loaded = load_chat_replay(stream_dir, stream_id="str_01ABC", tenant_id="ten_a")
    assert loaded is not None
    assert [(m.ts, m.user, m.text) for m in loaded.messages] == [
        (10.0, "alice", "first"),
        (12.5, "bob", "second"),
    ]


def test_save_sorts_by_timestamp(tmp_path: Path) -> None:
    stream_dir = tmp_path / "str_01ABC"
    out_of_order = _replay(
        (50.0, "c", "third"),
        (10.0, "a", "first"),
        (20.0, "b", "second"),
    )
    save_chat_replay(stream_dir, out_of_order)
    loaded = load_chat_replay(stream_dir, stream_id="str_01ABC", tenant_id="ten_a")
    assert loaded is not None
    assert [m.ts for m in loaded.messages] == [10.0, 20.0, 50.0]


def test_load_missing_returns_none(tmp_path: Path) -> None:
    assert load_chat_replay(tmp_path / "str_X", stream_id="str_X", tenant_id="ten_a") is None


def test_load_skips_blank_lines(tmp_path: Path) -> None:
    stream_dir = tmp_path / "str_01ABC"
    chat_replay_path(stream_dir).parent.mkdir(parents=True, exist_ok=True)
    chat_replay_path(stream_dir).write_text(
        '{"ts": 1.0, "user": "a", "text": "x"}\n\n'
        '{"ts": 2.0, "user": "b", "text": "y"}\n',
        encoding="utf-8",
    )
    loaded = load_chat_replay(stream_dir, stream_id="str_01ABC", tenant_id="ten_a")
    assert loaded is not None
    assert len(loaded.messages) == 2


def test_import_normalizes_external_jsonl(tmp_path: Path) -> None:
    """A user-provided JSONL gets imported into the canonical chat.jsonl
    location with stream_id + tenant_id stamped on it."""
    src = tmp_path / "external_chat.jsonl"
    src.write_text(
        '{"ts": 30.0, "user": "fan1", "text": "lol"}\n'
        '{"ts": 31.0, "user": "fan2", "text": "no way"}\n',
        encoding="utf-8",
    )
    stream_dir = tmp_path / "str_01ABC"
    replay = import_chat_replay(
        source=src,
        stream_dir=stream_dir,
        stream_id="str_01ABC",
        tenant_id="ten_a",
    )
    assert replay.stream_id == "str_01ABC"
    assert replay.tenant_id == "ten_a"
    assert chat_replay_path(stream_dir).exists()


def test_import_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="source not found"):
        import_chat_replay(
            source=tmp_path / "no_such_file.jsonl",
            stream_dir=tmp_path / "str_X",
            stream_id="str_X",
            tenant_id="ten_a",
        )


def test_import_bad_line_raises_with_line_number(tmp_path: Path) -> None:
    src = tmp_path / "bad.jsonl"
    src.write_text(
        '{"ts": 1.0, "user": "a", "text": "ok"}\n'
        '{"this is": "not a chat message"}\n',
        encoding="utf-8",
    )
    with pytest.raises(IngestError, match="line 2"):
        import_chat_replay(
            source=src,
            stream_dir=tmp_path / "str_X",
            stream_id="str_X",
            tenant_id="ten_a",
        )
