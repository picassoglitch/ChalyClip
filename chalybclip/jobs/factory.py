"""Factory — picks the JobDispatcher based on Settings."""

from __future__ import annotations

from chalybclip.errors import ChalybClipError
from chalybclip.settings import get_settings

from .base import JobDispatcher, PipelineRunner
from .in_process import InProcessJobDispatcher
from .modal import ModalJobDispatcher


def get_dispatcher(*, runner: PipelineRunner | None = None) -> JobDispatcher:
    """Return the configured job dispatcher.

    Args:
        runner: Required when `job_dispatcher == "in_process"` — the
            actual coroutine the dispatcher invokes. The Modal dispatcher
            uses it too, to build its in-process FALLBACK for sources
            that exist only on this box (`upload://`, `live://`).

    Defaults to "in_process" so existing deployments don't change behavior.
    """
    choice = (get_settings().job_dispatcher or "in_process").strip().lower()

    if choice == "in_process":
        if runner is None:
            raise ChalybClipError(
                "get_dispatcher(runner=...) is required for in-process dispatch"
            )
        return InProcessJobDispatcher(runner)

    if choice == "modal":
        fallback = InProcessJobDispatcher(runner) if runner is not None else None
        return ModalJobDispatcher(fallback=fallback)

    raise ChalybClipError(
        f"unknown job_dispatcher {choice!r}; expected one of "
        f"in_process / modal"
    )
