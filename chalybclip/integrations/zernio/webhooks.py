"""Inbound Zernio webhook verification + parsing.

Zernio POSTs us post-status events (published / failed / etc.) so we
don't have to poll. The transport auth IS the HMAC signature (there's
no cookie/session on a server-to-server callback), verified here
before we trust the body.

RISK / TODO: Zernio's exact signature scheme — the header name, the
bytes that get signed, and hex-vs-base64 encoding — is not yet pinned
against real traffic. We model the common shape (hex HMAC-SHA256 of
the raw request body, matching our own outbound `sign_payload` in
`chalybclip.webhooks.service`) behind the single `verify_zernio_signature`
function so swapping it later is a one-spot change. During rollout the
receiver logs the received header on mismatch so we can confirm the
real scheme, then enforce.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Any, Final

# Best-guess header Zernio sends the signature in. Confirm against
# real events; change here if it differs.
SIGNATURE_HEADER: Final = "X-Zernio-Signature"


def verify_zernio_signature(
    *,
    secret: str,
    body: bytes,
    signature_header: str | None,
) -> bool:
    """Return True iff `signature_header` is a valid HMAC-SHA256 (hex)
    of `body` under `secret`.

    Constant-time compare via `hmac.compare_digest`. A common variant
    prefixes the hex digest with `sha256=`; we tolerate that so a
    GitHub-style header still verifies. Returns False on any missing
    input rather than raising — the caller maps that to 403.
    """
    if not secret or not signature_header:
        return False
    provided = signature_header.strip()
    if provided.startswith("sha256="):
        provided = provided[len("sha256="):]
    expected = hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    # Compare as bytes: `compare_digest` on str raises TypeError for
    # non-ASCII input, which would turn a malformed/hostile header into an
    # unhandled 500 instead of the contract's False → 403.
    return hmac.compare_digest(expected.encode("ascii"), provided.encode("utf-8"))


def parse_post_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Zernio webhook body into the fields we care about.

    Tolerant of the envelope shape (top-level vs nested under `post`/
    `data`) since the exact contract is still being confirmed. Returns
    a flat dict; missing fields come back as None.
    """
    inner = payload
    for key in ("post", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            inner = nested
            break
    return {
        "event": payload.get("event") or payload.get("type"),
        "post_id": inner.get("_id") or inner.get("id") or inner.get("postId"),
        "status": inner.get("status"),
        "platform": inner.get("platform"),
        "platforms": inner.get("platforms"),
        "profile_id": inner.get("profileId"),
    }


# Display name of the hub's webhook on Zernio — the idempotency key
# for register_zernio_webhook (one hub == one webhook config).
WEBHOOK_NAME: Final = "ChalyClip Hub"


async def register_zernio_webhook(
    client: Any,
    *,
    url: str,
    secret: str,
    events: list[str] | None = None,
    name: str = WEBHOOK_NAME,
) -> dict[str, Any]:
    """Create-or-update the hub's webhook config on Zernio (idempotent).

    Finds an existing webhook by name (or exact URL) and updates it in
    place — also re-activating it if Zernio auto-disabled it after
    consecutive delivery failures — otherwise creates it. Returns
    {"action": "created"|"updated", "webhook_id": ..., "url": ...,
    "events": [...]}.

    `client` is a ZernioClient (typed as Any to keep this module free
    of the client import cycle; it only needs list/create/update).
    """
    if events is None:
        from .events import SUBSCRIBED_EVENTS

        events = list(SUBSCRIBED_EVENTS)

    existing = await client.list_webhooks()
    match = None
    for row in existing:
        if row.get("name") == name or row.get("url") == url:
            match = row
            break

    if match is not None:
        webhook_id = str(match.get("_id") or match.get("id") or "")
        await client.update_webhook(
            webhook_id,
            url=url,
            secret=secret,
            events=events,
            is_active=True,
        )
        return {
            "action": "updated",
            "webhook_id": webhook_id,
            "url": url,
            "events": events,
        }

    created = await client.create_webhook(
        name=name, url=url, secret=secret, events=events,
    )
    return {
        "action": "created",
        "webhook_id": str(created.get("_id") or created.get("id") or ""),
        "url": url,
        "events": events,
    }


__all__ = [
    "SIGNATURE_HEADER",
    "WEBHOOK_NAME",
    "parse_post_event",
    "register_zernio_webhook",
    "verify_zernio_signature",
]
