"""Modal web-endpoint polling protocol — shared client-side helper.

Modal's `fastapi_endpoint` for a long-running function answers the
initial POST with `303 See Other` + `Location: ?__modal_function_call_id=
fc-...`, and that polling URL keeps redirecting (302/303) until the
function completes (200 + JSON body) or fails (4xx/5xx). Both the whisper
transcribe provider and the pipeline job dispatcher speak this protocol;
this module is the single implementation of the follow-the-redirects loop.

Extracted from `chalybclip/transcribe/providers/modal_whisper.py` (Phase 2b)
— the behavior and the deadline-error message shape are unchanged.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import structlog

from chalybclip.errors import ChalybClipError

_log = structlog.get_logger(__name__)

POLL_REDIRECT_STATUSES = (302, 303, 307, 308)
"""HTTP redirect codes Modal uses to mean 'still running, try the
Location URL again later'. Treated uniformly."""


class ModalPollDeadlineError(ChalybClipError):
    """Raised when a Modal function call outlives the caller's polling
    deadline. NOTE: the remote function may still be running — the caller
    decides whether that means failure (whisper: yes, the pipeline is
    blocked on the result) or just lost tracking (job dispatcher: no,
    the run keeps writing its own step events)."""


async def poll_until_terminal(
    *,
    client: httpx.AsyncClient,
    initial: httpx.Response,
    deadline_s: float,
    poll_interval_s: float = 5.0,
    label: str = "modal",
    ref: str = "",
) -> httpx.Response:
    """Follow Modal's redirect-based polling protocol until the response
    is a terminal one (any non-redirect status).

    Behavior:
      * `initial` is the result of the POST. If it's already non-redirect
        (small/fast Modal function), return it immediately.
      * Otherwise GET the Location URL every `poll_interval_s` until a
        non-redirect response lands or `deadline_s` elapses (then
        `ModalPollDeadlineError`).

    Same return contract as a direct request: `raise_for_status()` is the
    caller's responsibility. `label`/`ref` only flavor logs + the deadline
    error message.
    """
    if initial.status_code not in POLL_REDIRECT_STATUSES:
        return initial

    deadline = time.monotonic() + max(0.0, deadline_s)
    current = initial
    polls = 0
    while current.status_code in POLL_REDIRECT_STATUSES:
        location = current.headers.get("location")
        if not location:
            # Redirect without a Location — protocol broken; treat the
            # response as terminal so the caller's raise_for_status
            # surfaces the unexpected 30x as a clear error.
            _log.warning(
                "modal_http.poll.no_location",
                label=label, ref=ref, status=current.status_code,
            )
            return current
        # Resolve the (possibly relative) Location against the response's
        # own request URL so trailing-slash + query-string cases stay
        # correct on Modal's hostnames. URL.join handles both absolute
        # and relative Locations (httpx.URL has no `base=` kwarg — the
        # code this was extracted from would have raised TypeError on
        # the first relative redirect).
        next_url = current.request.url.join(location)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModalPollDeadlineError(
                f"{label} poll deadline reached after {polls} polls "
                f"(deadline={deadline_s:.0f}s)"
            )

        # Brief backoff before the next poll. Cap by remaining deadline so
        # the final poll fires at most just-before the deadline rather
        # than oversleeping past it.
        await asyncio.sleep(min(poll_interval_s, remaining))
        polls += 1
        if polls == 1 or polls % 12 == 0:
            # Log first poll + periodically so prod logs show progress
            # without spamming on every poll.
            _log.info(
                "modal_http.poll",
                label=label, ref=ref, poll_n=polls,
                next_url_host=str(next_url.host),
            )
        current = await client.get(str(next_url))
    return current


__all__ = [
    "POLL_REDIRECT_STATUSES",
    "ModalPollDeadlineError",
    "poll_until_terminal",
]
