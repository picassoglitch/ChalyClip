"""Webhook dispatcher — HMAC-signed delivery of event rows to subscriber URLs.

Per subscription:
  1. Read events newer than `last_dispatch_ts`, filtered by `types_json`.
  2. POST each event as JSON to `url` with two headers:
       X-Chalybclip-Signature: hex(hmac_sha256(secret, body))
       X-Chalybclip-Timestamp: iso-8601 of the request build time
  3. On 2xx -> bump `last_dispatch_ts` to that event's ts (and reset
     `failure_count`).
  4. On any error (4xx, 5xx, network) -> increment `failure_count`. After
     `MAX_FAILURES_BEFORE_DISABLE` consecutive failures, flip the
     subscription `status` to `disabled`.

Phase 2 keeps the dispatcher single-flight per subscription (no parallel
delivery to the same URL) - simpler to reason about and good enough for
the dashboard's expected fanout.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
import structlog

from chalybclip.db import (
    Database,
    EventsRepo,
    WebhookSubscriptionsRepo,
)
from chalybclip.db.models import Event, WebhookSubscription
from chalybclip.tenancy import bound_tenant
from chalybclip.webhooks.url_safety import (
    UnsafeWebhookUrlError,
    assert_webhook_url_safe,
)

_log = structlog.get_logger(__name__)

HMAC_HEADER = "X-Chalybclip-Signature"
SIGNED_TS_HEADER = "X-Chalybclip-Timestamp"

# After this many consecutive POST failures we disable the subscription.
# The dashboard surfaces it; the operator re-enables after fixing the URL.
MAX_FAILURES_BEFORE_DISABLE = 10


@dataclass(frozen=True)
class DispatchOutcome:
    """Roll-up of one drain pass."""

    delivered: int
    failed: int
    disabled: int  # subscriptions newly flipped to status=disabled


def sign_payload(*, secret: str, body: bytes) -> str:
    """Return hex HMAC-SHA256 of `body` under `secret`."""
    return hmac.new(
        secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _events_after(events: list[Event], cutoff: str | None) -> list[Event]:
    """Return events strictly after `cutoff` (the subscription's last_dispatch_ts).

    Events are sorted DESC by ts in the repo; we re-sort ASC here so we
    deliver in chronological order.
    """
    asc = sorted(events, key=lambda e: e.ts)
    if cutoff is None:
        return asc
    return [e for e in asc if e.ts > cutoff]


def _matches_types(event: Event, types: list[str]) -> bool:
    """`types == []` means "all events"; otherwise prefix-or-exact match."""
    if not types:
        return True
    for t in types:
        if t.endswith("*") and event.type.startswith(t[:-1]):
            return True
        if event.type == t:
            return True
    return False


async def run_webhook_dispatch(
    tenant_id: str,
    db: Database,
    *,
    http: httpx.AsyncClient | None = None,
    clock: Callable[[], _dt.datetime] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    max_events_per_sub: int = 200,
) -> DispatchOutcome:
    """Drain pending events for every active webhook subscription on `tenant_id`.

    Returns roll-up counts. Per-event delivery failures bump
    `failure_count`; successive failures disable the subscription.
    """
    own_http = http is None
    http_client = http or httpx.AsyncClient(timeout=10.0)
    delivered = 0
    failed = 0
    disabled = 0
    try:
        with bound_tenant(tenant_id):
            subs_repo = WebhookSubscriptionsRepo(db)
            events_repo = EventsRepo(db)

            subs = await subs_repo.list_for_tenant(status="active")
            if not subs:
                return DispatchOutcome(delivered=0, failed=0, disabled=0)

            # Pull a generous window of recent events, then filter per
            # subscription. Phase 1's events table is tenant-indexed so
            # this is one query regardless of subscription count.
            all_events = await events_repo.list_for_tenant(limit=max_events_per_sub)

            for sub in subs:
                pending = [
                    e
                    for e in _events_after(all_events, sub.last_dispatch_ts)
                    if _matches_types(e, sub.types)
                ]
                if not pending:
                    continue
                d, f, dis = await _deliver_one_sub(
                    sub=sub,
                    events=pending,
                    db=db,
                    http=http_client,
                    subs_repo=subs_repo,
                )
                delivered += d
                failed += f
                disabled += dis
    finally:
        if own_http:
            await http_client.aclose()

    if delivered or failed or disabled:
        _log.info(
            "webhook_drain",
            tenant_id=tenant_id,
            delivered=delivered,
            failed=failed,
            disabled=disabled,
        )
    return DispatchOutcome(delivered=delivered, failed=failed, disabled=disabled)


async def _deliver_one_sub(
    *,
    sub: WebhookSubscription,
    events: list[Event],
    db: Database,
    http: httpx.AsyncClient,
    subs_repo: WebhookSubscriptionsRepo,
) -> tuple[int, int, int]:
    """Deliver events to one subscription. Returns (delivered, failed, disabled)."""
    delivered = 0
    failed = 0
    last_success_ts: str | None = None

    # SSRF re-check at send time: DNS rebinding could swap a previously
    # public-resolving hostname for a private one between registration
    # and now. If it does, disable this subscription rather than letting
    # it fan out to internal services. Done once per drain pass, not per
    # event, since the URL is fixed for the subscription. `allow_unresolvable`
    # so a transiently-down DNS doesn't trip the safety net — the existing
    # httpx failure handling covers connectivity issues; we only care about
    # the "resolved to private" case here.
    try:
        assert_webhook_url_safe(sub.url, allow_unresolvable=True)
    except UnsafeWebhookUrlError as e:
        _log.warning(
            "webhook_url_unsafe_at_dispatch",
            sub_id=sub.id,
            url=sub.url,
            error=str(e),
        )
        await _disable_subscription(db, sub.id)
        return delivered, failed, 1

    for event in events:
        body = _serialize(event)
        sig = sign_payload(secret=sub.secret, body=body)
        try:
            resp = await http.post(
                sub.url,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    HMAC_HEADER: sig,
                    SIGNED_TS_HEADER: _utc_now().isoformat(),
                },
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
            failed += 1
            count = await subs_repo.record_failure(sub.id)
            _log.info(
                "webhook_failed",
                sub_id=sub.id,
                url=sub.url,
                error=str(e),
                failure_count=count,
            )
            if count >= MAX_FAILURES_BEFORE_DISABLE:
                await _disable_subscription(db, sub.id)
                return delivered, failed, 1
            return delivered, failed, 0

        if 200 <= resp.status_code < 300:
            delivered += 1
            last_success_ts = event.ts
        else:
            failed += 1
            count = await subs_repo.record_failure(sub.id)
            if count >= MAX_FAILURES_BEFORE_DISABLE:
                await _disable_subscription(db, sub.id)
                if last_success_ts is not None:
                    await subs_repo.record_dispatch(sub.id, ts=last_success_ts)
                return delivered, failed, 1
            break  # don't keep trying once one fails this pass

    if last_success_ts is not None:
        await subs_repo.record_dispatch(sub.id, ts=last_success_ts)
    return delivered, failed, 0


async def _disable_subscription(db: Database, sub_id: str) -> None:
    """Mark the subscription disabled - dashboard surfaces a "reconnect" CTA."""
    conn = await db.connect()
    await conn.execute(
        "UPDATE webhook_subscriptions SET status = 'disabled' WHERE id = ?",
        (sub_id,),
    )
    await conn.commit()


def _serialize(event: Event) -> bytes:
    """Stable JSON shape for HMAC + delivery."""
    return json.dumps(
        {
            "id": event.id,
            "tenant_id": event.tenant_id,
            "type": event.type,
            "payload": event.payload,
            "ts": event.ts,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
