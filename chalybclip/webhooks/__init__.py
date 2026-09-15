"""Webhook dispatch — HMAC-signed event delivery to subscriber URLs.

The dispatcher reads new event rows since each subscription's
`last_dispatch_ts`, filters by subscribed types, signs payloads with
HMAC-SHA256(secret), POSTs to `url`, and bumps either `last_dispatch_ts`
on success or `failure_count` on failure.

Two callers:
    * `chalybclip webhooks send --tenant <id>` for one-shot drains.
    * The FastAPI lifespan task wakes every 30s and runs the same drain.
"""

from .service import (
    HMAC_HEADER,
    SIGNED_TS_HEADER,
    DispatchOutcome,
    run_webhook_dispatch,
    sign_payload,
)
from .url_safety import UnsafeWebhookUrlError, assert_webhook_url_safe

__all__ = [
    "HMAC_HEADER",
    "SIGNED_TS_HEADER",
    "DispatchOutcome",
    "UnsafeWebhookUrlError",
    "assert_webhook_url_safe",
    "run_webhook_dispatch",
    "sign_payload",
]
