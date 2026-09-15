"""structlog configuration shared by the CLI, FastAPI, and workers.

Per CLAUDE.md: structured fields, not f-string concatenation. Logs go to
stderr by default so a `--json` CLI command can stream the manifest to
stdout without log noise interleaving.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Idempotent process-wide logger setup. Call once at entry."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,
        format="%(message)s",
        force=True,
    )

    # Admin log tracker — mirror WARNING+ records into the in-memory
    # ring buffer behind /dashboard/_health/logs. Two hooks because the
    # two log worlds don't meet: the stdlib handler sees uvicorn +
    # third-party records, the structlog processor (added below, before
    # the renderer) sees our own app logs, which go straight to stderr
    # via PrintLoggerFactory and never touch stdlib logging.
    from chalybclip.logbuffer import StdlibCaptureHandler, structlog_capture

    capture_handler = StdlibCaptureHandler()
    capture_handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(capture_handler)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog_capture,
    ]
    if fmt.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger; pass `__name__` from caller modules.

    Typed as `Any` because structlog's runtime logger class varies with the
    configured `wrapper_class`; pinning a specific class would lie to mypy.
    """
    return structlog.get_logger(name) if name else structlog.get_logger()


def reset_for_tests() -> None:
    """Allow tests to reconfigure log level by clearing the one-shot guard."""
    global _CONFIGURED
    _CONFIGURED = False
