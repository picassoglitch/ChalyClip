"""Admin log tracker — ring buffer capture + /dashboard/_health/logs.

The buffer is a process-wide singleton, so every test clears it first;
capture tests drive the structlog processor / stdlib handler directly
instead of reconfiguring global logging (configure_logging is one-shot
per process and other tests depend on its state).
"""

from __future__ import annotations

import logging

import pytest

from chalybclip.logbuffer import (
    MAX_FIELD_CHARS,
    LogEntry,
    LogRingBuffer,
    StdlibCaptureHandler,
    get_log_buffer,
    structlog_capture,
)

from .conftest import auth


@pytest.fixture(autouse=True)
def _clean_buffer():
    get_log_buffer().clear()
    yield
    get_log_buffer().clear()


# ---------- capture hooks ----------


def test_structlog_processor_captures_warning_and_error() -> None:
    for method in ("warning", "error"):
        out = structlog_capture(
            None,
            method,
            {
                "event": f"render.{method}",
                "timestamp": "2026-07-06T12:00:00+00:00",
                "level": method,
                "clip_id": "clp_x",
            },
        )
        # Processor passes the event through unchanged.
        assert out["event"] == f"render.{method}"

    entries = get_log_buffer().snapshot()
    assert [e.event for e in entries] == ["render.error", "render.warning"]
    assert entries[0].level == "error"
    assert entries[0].fields == {"clip_id": "clp_x"}


def test_structlog_processor_ignores_info() -> None:
    structlog_capture(None, "info", {"event": "quiet", "level": "info"})
    assert get_log_buffer().snapshot() == []


def test_structlog_exception_maps_to_error_level() -> None:
    structlog_capture(None, "exception", {"event": "boom"})
    assert get_log_buffer().snapshot()[0].level == "error"


def test_field_values_are_truncated() -> None:
    structlog_capture(
        None, "warning", {"event": "big", "cmd": "x" * (MAX_FIELD_CHARS * 2)}
    )
    v = get_log_buffer().snapshot()[0].fields["cmd"]
    assert len(v) == MAX_FIELD_CHARS


def test_stdlib_handler_captures_warning_with_exception() -> None:
    handler = StdlibCaptureHandler()
    logger = logging.getLogger("test.stdlib.capture")
    record = logger.makeRecord(
        "test.stdlib.capture",
        logging.ERROR,
        __file__,
        1,
        "upload failed: %s",
        ("disk full",),
        (ValueError, ValueError("disk full"), None),
    )
    handler.emit(record)

    (entry,) = get_log_buffer().snapshot()
    assert entry.level == "error"
    assert entry.logger == "test.stdlib.capture"
    assert entry.event == "upload failed: disk full"
    assert "disk full" in entry.fields["exception"]


def test_ring_buffer_bounded_and_newest_first() -> None:
    buf = LogRingBuffer(maxlen=3)
    for i in range(5):
        buf.append(
            LogEntry(ts=f"2026-07-06T12:00:0{i}+00:00", level="warning",
                     logger="t", event=f"e{i}")
        )
    events = [e.event for e in buf.snapshot()]
    assert events == ["e4", "e3", "e2"]


def test_counts_bucket_by_recency_and_level() -> None:
    from datetime import UTC, datetime, timedelta

    buf = LogRingBuffer()
    now = datetime.now(UTC)
    buf.append(LogEntry(ts=now.isoformat(), level="error", logger="t", event="fresh"))
    buf.append(
        LogEntry(ts=(now - timedelta(hours=5)).isoformat(), level="warning",
                 logger="t", event="older")
    )
    buf.append(
        LogEntry(ts=(now - timedelta(days=3)).isoformat(), level="error",
                 logger="t", event="ancient")
    )
    c = buf.counts()
    assert c["errors_1h"] == 1
    assert c["errors_24h"] == 1
    assert c["warnings_24h"] == 1
    assert c["warnings_1h"] == 0
    assert c["total"] == 3


# ---------- route ----------


@pytest.fixture
def _admin(tenants, monkeypatch):
    """Make alice an admin for the duration of the test."""
    from chalybclip.settings import get_settings

    monkeypatch.setenv("CHALYBCLIP_ADMIN_TENANT_IDS", tenants["alice"]["id"])
    get_settings.cache_clear()
    yield tenants["alice"]
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_logs_health_404_for_non_admin(client, tenants) -> None:
    r = await client.get(
        "/dashboard/_health/logs", headers=auth(tenants["bob"]["token"])
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_logs_health_renders_captured_entries(client, _admin) -> None:
    structlog_capture(
        None,
        "error",
        {"event": "publish.sweep.failed", "platform": "youtube"},
    )
    r = await client.get("/dashboard/_health/logs", headers=auth(_admin["token"]))
    assert r.status_code == 200
    assert "publish.sweep.failed" in r.text
    assert "youtube" in r.text


@pytest.mark.asyncio
async def test_logs_health_level_and_text_filters(client, _admin) -> None:
    structlog_capture(None, "warning", {"event": "render.slow"})
    structlog_capture(None, "error", {"event": "publish.failed"})

    r = await client.get(
        "/dashboard/_health/logs",
        params={"level": "error"},
        headers=auth(_admin["token"]),
    )
    assert "publish.failed" in r.text
    assert "render.slow" not in r.text

    r = await client.get(
        "/dashboard/_health/logs",
        params={"q": "render"},
        headers=auth(_admin["token"]),
    )
    assert "render.slow" in r.text
    assert "publish.failed" not in r.text


@pytest.mark.asyncio
async def test_logs_health_empty_state(client, _admin) -> None:
    r = await client.get("/dashboard/_health/logs", headers=auth(_admin["token"]))
    assert r.status_code == 200
    assert "Nothing captured yet" in r.text
