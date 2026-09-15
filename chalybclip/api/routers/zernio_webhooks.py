"""Inbound Zernio webhook receiver — the hub's event backbone (phase 2).

Zernio POSTs events here (post status, account changes, comments,
messages, external posts) so we don't poll. There's no tenant cookie
on a server-to-server callback — the transport auth IS the HMAC
signature, verified before we trust the body. This router is exempt
from the bearer-cookie middleware (see `/api/webhooks/` in
auth._PUBLIC_PREFIXES); the handler enforces its own signature check.

Contract (per Zernio's spec):
  - signature: X-Zernio-Signature = lowercase hex HMAC-SHA256 of the
    raw body keyed by our webhook secret
  - delivery is at-least-once; payload.id is the stable dedup key
  - ACK 2xx within 5s or Zernio retries (exponential backoff, 7 tries)

So the handler does the MINIMUM inline — verify, dedup-insert, 200 —
and hands the real work (tenant resolution, publish-status updates,
fan-out to subscriber webhooks) to a background task via
chalybclip.integrations.zernio.events.process_zernio_event.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from chalybclip.db import Database, ZernioEventsRepo
from chalybclip.ids import new_id
from chalybclip.integrations.zernio.events import (
    extract_profile_id,
    process_zernio_event,
)
from chalybclip.integrations.zernio.webhooks import (
    SIGNATURE_HEADER,
    verify_zernio_signature,
)
from chalybclip.settings import get_settings

from ..deps import get_db

_log = logging.getLogger("chalybclip.api.zernio_webhooks")

router = APIRouter(prefix="/api/webhooks", tags=["zernio"], include_in_schema=False)

# Zernio repeats payload.id in this header; used as fallback when the
# body parses but carries no id (defensive — the spec requires one).
EVENT_ID_HEADER = "X-Zernio-Event-Id"


@router.post("/zernio")
async def zernio_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_db),
) -> JSONResponse:
    """Receive, verify, dedup and ACK one Zernio webhook event.

    503 when the verification secret isn't configured (we refuse to
    trust an event we can't verify), 403 on signature mismatch, 400 on
    a non-JSON body. A duplicate event_id ACKs 200 immediately (the
    first delivery already processed it). Fresh events are stored
    verbatim and processed AFTER the response goes out.
    """
    settings = get_settings()
    secret = (settings.zernio_webhook_secret or "").strip()
    if not secret:
        raise HTTPException(
            status_code=503,
            detail=(
                "CHALYBCLIP_ZERNIO_WEBHOOK_SECRET is not configured; "
                "cannot verify inbound Zernio webhooks."
            ),
        )

    body = await request.body()
    provided = request.headers.get(SIGNATURE_HEADER)
    if not verify_zernio_signature(
        secret=secret, body=body, signature_header=provided,
    ):
        # Log the received header (NOT the secret) so we can confirm
        # Zernio's real signature scheme during rollout, then enforce.
        _log.warning(
            "zernio.webhook.bad_signature header=%r len=%d",
            provided, len(body),
        )
        raise HTTPException(status_code=403, detail="invalid signature")

    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(status_code=400, detail="body is not JSON") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body is not a JSON object")

    raw_id = payload.get("id") or request.headers.get(EVENT_ID_HEADER)
    event_id = raw_id if isinstance(raw_id, str) and raw_id else new_id("zev")
    raw_type = payload.get("event") or payload.get("type")
    event_type = raw_type if isinstance(raw_type, str) and raw_type else "unknown"

    inserted = await ZernioEventsRepo(db).insert_dedup(
        event_id=event_id,
        type=event_type,
        payload=body.decode("utf-8", errors="replace"),
        profile_id=extract_profile_id(payload),
    )
    if not inserted:
        # At-least-once redelivery — already stored (and processed or
        # in-flight). ACK so Zernio stops retrying.
        return JSONResponse({"ok": True, "duplicate": True})

    _log.info(
        "zernio.webhook.received event_id=%s type=%s", event_id, event_type,
    )
    # Process AFTER the 200 goes out — Zernio wants the ACK within 5s
    # and processing does DB writes + outbound fan-out HTTP.
    background_tasks.add_task(process_zernio_event, db, event_id)
    return JSONResponse({"ok": True})


__all__ = ["router"]
