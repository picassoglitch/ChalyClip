"""In-memory ring buffer of recent WARNING+ log records for the admin
log tracker (`/dashboard/_health/logs`).

Two capture hooks feed the same buffer:

  * `structlog_capture` — a structlog processor inserted before the
    renderer in `chalybclip.logging.configure_logging`. Our app logs go
    through structlog's PrintLoggerFactory straight to stderr and never
    touch stdlib logging, so a logging.Handler alone would miss them.
  * `StdlibCaptureHandler` — attached to the root stdlib logger for
    everything that does NOT go through structlog (uvicorn errors,
    third-party libraries).

Deliberately in-memory only: this is a "what went wrong recently"
operator surface, not an audit log. Railway keeps the durable copy of
stdout/stderr; a process restart clearing the buffer is acceptable and
the page says since-when it has data. Keeping it out of the DB also
means a database outage still shows up here instead of taking the
error tracker down with it.

Module-level singleton on purpose (mirrors the `_CONFIGURED` guard in
chalybclip/logging.py): log capture is process-wide infrastructure wired
once at entry, not a per-request dependency.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

# Bounded so a log storm can't eat the box. ~1000 entries of a few
# hundred bytes each is single-digit MB worst case.
MAX_ENTRIES = 1000
# Per-field value cap. Log fields can embed whole ffmpeg commands or
# stack traces; the tracker only needs enough to identify the problem —
# the full line is still on stderr/Railway.
MAX_FIELD_CHARS = 500
MAX_FIELDS = 20

_CAPTURED_LEVELS = {"warning", "error", "critical", "exception"}


@dataclass
class LogEntry:
    """One captured WARNING+ record, already stringified and truncated."""

    ts: str  # ISO-8601 UTC
    level: str  # "warning" | "error" | "critical"
    logger: str
    event: str
    fields: dict[str, str] = field(default_factory=dict)


class LogRingBuffer:
    """Thread-safe fixed-size buffer. Appends come from any thread
    (stdlib handler) or the event loop (structlog processor); reads come
    from the dashboard request handler."""

    def __init__(self, maxlen: int = MAX_ENTRIES) -> None:
        self._entries: deque[LogEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.started_at: str = datetime.now(UTC).isoformat()

    def append(self, entry: LogEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def snapshot(self) -> list[LogEntry]:
        """Newest first."""
        with self._lock:
            return list(reversed(self._entries))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        self.started_at = datetime.now(UTC).isoformat()

    def counts(self) -> dict[str, int]:
        """Error/warning tallies for the summary chips: last hour and
        last 24 h, computed against entry timestamps (ISO strings sort
        lexicographically, same trick the queue-health page uses)."""
        now = datetime.now(UTC)
        hour_cutoff = (now - timedelta(hours=1)).isoformat()
        day_cutoff = (now - timedelta(hours=24)).isoformat()
        out = {
            "errors_1h": 0,
            "warnings_1h": 0,
            "errors_24h": 0,
            "warnings_24h": 0,
            "total": 0,
        }
        with self._lock:
            for e in self._entries:
                out["total"] += 1
                is_error = e.level in ("error", "critical")
                if e.ts >= day_cutoff:
                    out["errors_24h" if is_error else "warnings_24h"] += 1
                if e.ts >= hour_cutoff:
                    out["errors_1h" if is_error else "warnings_1h"] += 1
        return out


_buffer = LogRingBuffer()


def get_log_buffer() -> LogRingBuffer:
    return _buffer


def _clip(value: Any) -> str:
    s = str(value)
    return s if len(s) <= MAX_FIELD_CHARS else s[: MAX_FIELD_CHARS - 1] + "…"


def structlog_capture(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor: siphon WARNING+ events into the buffer,
    pass the event through unchanged. Never raises — a broken capture
    must not break logging itself."""
    try:
        if method_name in _CAPTURED_LEVELS:
            fields: dict[str, str] = {}
            for k, v in event_dict.items():
                if k in ("event", "timestamp", "level", "logger"):
                    continue
                if len(fields) >= MAX_FIELDS:
                    fields["…"] = "(more fields truncated)"
                    break
                fields[str(k)] = _clip(v)
            _buffer.append(
                LogEntry(
                    ts=str(
                        event_dict.get("timestamp")
                        or datetime.now(UTC).isoformat()
                    ),
                    level="error" if method_name == "exception" else method_name,
                    logger=str(
                        event_dict.get("logger")
                        or getattr(logger, "name", "")
                        or "structlog"
                    ),
                    event=str(event_dict.get("event", "")),
                    fields=fields,
                )
            )
    except Exception:  # capture is best-effort observability
        pass
    return event_dict


class StdlibCaptureHandler(logging.Handler):
    """Root-logger handler for the non-structlog world (uvicorn,
    libraries). WARNING+ only, set via the handler level at attach."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            fields: dict[str, str] = {}
            if record.exc_info and record.exc_info[1] is not None:
                fields["exception"] = _clip(repr(record.exc_info[1]))
            _buffer.append(
                LogEntry(
                    ts=datetime.fromtimestamp(record.created, UTC).isoformat(),
                    level=record.levelname.lower(),
                    logger=record.name,
                    event=_clip(record.getMessage()),
                    fields=fields,
                )
            )
        except Exception:  # never let capture break logging
            pass


__all__ = [
    "LogEntry",
    "LogRingBuffer",
    "StdlibCaptureHandler",
    "get_log_buffer",
    "structlog_capture",
]
