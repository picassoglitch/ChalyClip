"""VOD ingest module — public surface."""

from .chat_replay import (
    chat_replay_path,
    import_chat_replay,
    load_chat_replay,
    save_chat_replay,
)
from .models import ChatMessage, ChatReplay, Platform, Stream
from .service import (
    detect_platform,
    ingest_uploaded,
    ingest_vod,
    is_ffmpeg_available,
    load_stream,
)

__all__ = [
    "ChatMessage",
    "ChatReplay",
    "Platform",
    "Stream",
    "chat_replay_path",
    "detect_platform",
    "import_chat_replay",
    "ingest_uploaded",
    "ingest_vod",
    "is_ffmpeg_available",
    "load_chat_replay",
    "load_stream",
    "save_chat_replay",
]
