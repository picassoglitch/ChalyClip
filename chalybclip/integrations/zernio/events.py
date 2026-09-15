"""Zernio webhook event processing — the hub's event backbone.

The receiver (routers/zernio_webhooks.py) verifies + dedup-inserts the
raw event and ACKs fast; THIS module does the actual work in a
background task:

  1. Resolve the tenant: profileId → tenants.zernio_profile_id, or the
     post id → zernio_publishes (posts fired through the hub).
  2. Update local state: post.* events feed zernio_publishes.status so
     the dashboard/internal API read live status without polling.
     account.* needs no local write — connection chips read Zernio
     live (more accurate than a mirror). post.external.*, comment.*,
     message.* stay in zernio_events verbatim; the calendar (phase 8)
     and inbox (phase 9) stores consume them from there.
  3. Fan out: record a `zernio.<type>` row in the events table and
     drain the tenant's webhook subscriptions (existing HMAC-signed
     dispatcher) so ChalybOBS / Chalyb engines get the relay.

Everything here is idempotent: a redelivery never reaches processing
(dedup at insert), and a re-run of process_event on an already-
processed row is a no-op.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from chalybclip.db import (
    Database,
    EventsRepo,
    HubPublishJobsRepo,
    TenantsRepo,
    ZernioAutoRetriesRepo,
    ZernioCalendarRepo,
    ZernioEventsRepo,
    ZernioInboxRepo,
    ZernioPublishesRepo,
)
from chalybclip.tenancy import bound_tenant

_log = logging.getLogger("chalybclip.integrations.zernio.events")

# Strong refs to in-flight auto-retry tasks — asyncio.create_task only
# keeps a weak reference, so a fire-and-forget task can be GC'd mid-flight.
_auto_retry_tasks: set[Any] = set()

# Post-level events → the status we persist on zernio_publishes.
# The payload's own post.status wins when present; this is the
# fallback derived from the event name.
POST_EVENT_STATUS: dict[str, str] = {
    "post.scheduled": "scheduled",
    "post.published": "published",
    "post.failed": "failed",
    "post.partial": "partial",
    "post.cancelled": "cancelled",
    "post.recycled": "published",
}

# Events the hub subscribes to on Zernio (register_zernio_webhook).
# Only enum-valid names from the OpenAPI spec — conversation.started
# is documented in prose but absent from the subscription enum, so we
# can't subscribe to it (we still process it if it ever arrives).
SUBSCRIBED_EVENTS: tuple[str, ...] = (
    "post.scheduled",
    "post.published",
    "post.failed",
    "post.partial",
    "post.cancelled",
    "post.external.created",
    "post.external.updated",
    "post.external.deleted",
    "account.connected",
    "account.disconnected",
    "comment.received",
    "message.received",
    "message.sent",
)


def extract_profile_id(payload: dict[str, Any]) -> str | None:
    """Pull the Zernio profileId out of any documented payload shape.

    account.* carries it at account.profileId; other shapes have used a
    top-level or post-nested profileId. Tolerates the nested-object
    form ({_id: ...}) — same regression the accounts list had."""
    candidates: list[Any] = [payload.get("profileId")]
    for key in ("account", "post", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get("profileId"))
    for c in candidates:
        if isinstance(c, dict):
            c = c.get("_id") or c.get("id")
        if isinstance(c, str) and c:
            return c
    return None


def _extract_post(payload: dict[str, Any]) -> dict[str, Any] | None:
    post = payload.get("post")
    return post if isinstance(post, dict) else None


def _event_summary(type_: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Small primitive-typed payload for the fan-out event row.

    Ids + statuses only — never message/comment text (subscribers that
    need the full body fetch it via the API; our events table and the
    relay bodies stay free of user content)."""
    out: dict[str, Any] = {"zernio_event": type_}
    post = _extract_post(payload)
    if post:
        out["post_id"] = post.get("id") or post.get("_id")
        out["status"] = post.get("status")
    account = payload.get("account")
    if isinstance(account, dict):
        out["account_id"] = account.get("accountId")
        out["platform"] = account.get("platform")
    profile_id = extract_profile_id(payload)
    if profile_id:
        out["profile_id"] = profile_id
    return {k: v for k, v in out.items() if v is not None}


async def process_zernio_event(db: Database, event_id: str) -> None:
    """Process ONE stored webhook event. Safe to re-run; never raises
    (a webhook background task must not take the worker down)."""
    try:
        await _process(db, event_id)
    except Exception as e:
        _log.warning(
            "zernio.event.process_failed event_id=%s err=%s", event_id, e,
        )


async def _process(db: Database, event_id: str) -> None:
    events_repo = ZernioEventsRepo(db)
    row = await events_repo.get(event_id)
    if row is None or row.processed:
        return

    try:
        payload = json.loads(row.payload)
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        # Malformed but stored — mark processed so the sweep doesn't
        # spin on it; the raw body stays for diagnosis.
        await events_repo.mark_processed(event_id)
        return

    # --- tenant resolution ---
    tenant_id = row.tenant_id
    profile_id = row.profile_id or extract_profile_id(payload)
    if tenant_id is None and profile_id:
        tenant = await TenantsRepo(db).find_by_zernio_profile(profile_id)
        tenant_id = tenant.id if tenant else None

    # --- local state: post.* feeds the publish record ---
    post = _extract_post(payload)
    post_id = None
    if post:
        pid = post.get("id") or post.get("_id")
        post_id = pid if isinstance(pid, str) and pid else None
    if post_id and (
        row.type in POST_EVENT_STATUS or row.type.startswith("post.platform.")
    ):
        status = post.get("status") if post else None
        if not isinstance(status, str) or not status:
            status = POST_EVENT_STATUS.get(row.type)
        platforms = post.get("platforms") if post else None
        platforms_json = (
            json.dumps(platforms) if isinstance(platforms, list) else None
        )
        # Dashboard publishes (zernio_publishes) and internal-API jobs
        # (hub_publish_jobs) both key on the Zernio post id; a post
        # only ever lives in one of them, the other update no-ops.
        pub = await ZernioPublishesRepo(db).get_by_post_id(post_id)
        if pub is not None:
            tenant_id = tenant_id or pub.tenant_id
        elif tenant_id is None:
            tenant_id = await HubPublishJobsRepo(db).get_tenant_for_post(post_id)
        if status:
            if pub is not None:
                await ZernioPublishesRepo(db).set_status(
                    post_id, status=status, platforms_json=platforms_json,
                )
            await HubPublishJobsRepo(db).set_status_by_post_id(
                post_id, status=status, platforms_json=platforms_json,
            )

    # --- external posts feed the unified calendar (phase 8) ---
    # No profileId on these — keyed by the social account id, resolved
    # to a tenant at read time. created/updated upsert; deleted flags.
    if row.type.startswith("post.external.") and post:
        await _process_external_post(db, row.type, post)

    # --- inbox: comments + DMs feed the inbox store (phase 9) ---
    if row.type == "comment.received":
        await _process_comment(db, payload)
    elif row.type in ("message.received", "message.sent"):
        await _process_message(db, payload)
    elif row.type == "conversation.started":
        await _process_conversation(db, payload)
    elif row.type.startswith("whatsapp.number."):
        await _process_whatsapp_number(db, row.type, payload)

    # --- auto-retry once on a transient failure ---
    # post.failed / post.partial with ALL failed platforms transient
    # (platform/infra/velocity) gets ONE automatic retry after a delay.
    # The ledger claim is the once-only guard against redeliveries.
    if post_id and row.type in ("post.failed", "post.partial") and post:
        from .errors import post_is_auto_retryable

        if post_is_auto_retryable(post):
            await _maybe_auto_retry(db, post_id=post_id, tenant_id=tenant_id)

    # --- community announce-on-publish (phase 11) ---
    if post_id and row.type == "post.published" and tenant_id and post:
        await _maybe_notify_community(
            db, tenant_id=tenant_id, post_id=post_id, post=post,
        )

    # --- fan-out: event row + drain the tenant's webhook subscriptions ---
    if tenant_id:
        try:
            with bound_tenant(tenant_id):
                await EventsRepo(db).emit(
                    type=f"zernio.{row.type}",
                    payload=_event_summary(row.type, payload),
                )
            from chalybclip.webhooks import run_webhook_dispatch

            await run_webhook_dispatch(tenant_id, db)
        except Exception as e:
            _log.warning(
                "zernio.event.fanout_failed event_id=%s tenant=%s err=%s",
                event_id, tenant_id, e,
            )
    else:
        _log.info(
            "zernio.event.unresolved event_id=%s type=%s profile_id=%s",
            event_id, row.type, profile_id,
        )

    await events_repo.mark_processed(event_id, tenant_id=tenant_id)


async def _process_external_post(
    db: Database, event_type: str, post: dict[str, Any]
) -> None:
    """Upsert / delete an external post into the calendar store.

    `post.id` is the platform-native id, `post.accountId` the Zernio
    social account (the tenant link). Tolerant of missing fields — a
    sparse payload still records the entry."""
    post_id = post.get("id") or post.get("_id")
    account_id = post.get("accountId")
    if not (isinstance(post_id, str) and post_id and isinstance(account_id, str)):
        return
    repo = ZernioCalendarRepo(db)
    if event_type == "post.external.deleted":
        deleted_at = post.get("deletedAt")
        await repo.mark_deleted(
            account_id=account_id,
            post_id=post_id,
            deleted_at=deleted_at if isinstance(deleted_at, str) else None,
        )
        return
    await repo.upsert(
        account_id=account_id,
        post_id=post_id,
        platform=post.get("platform") if isinstance(post.get("platform"), str) else None,
        content=post.get("content") if isinstance(post.get("content"), str) else None,
        url=post.get("url") if isinstance(post.get("url"), str) else None,
        thumbnail_url=(
            post.get("thumbnailUrl")
            if isinstance(post.get("thumbnailUrl"), str) else None
        ),
        media_type=(
            post.get("mediaType") if isinstance(post.get("mediaType"), str) else None
        ),
        published_at=(
            post.get("publishedAt")
            if isinstance(post.get("publishedAt"), str) else None
        ),
    )


def _s(value: Any) -> str | None:
    """A string value or None — keeps the inbox writes type-clean."""
    return value if isinstance(value, str) and value else None


async def _process_comment(db: Database, payload: dict[str, Any]) -> None:
    """comment.received → store the comment + seed the author as a
    potential contact (feeds phase 10)."""
    comment = payload.get("comment")
    account = payload.get("account")
    if not isinstance(comment, dict) or not isinstance(account, dict):
        return
    account_id = _s(account.get("id"))
    comment_id = _s(comment.get("id"))
    if not (account_id and comment_id):
        return
    _author = comment.get("author")
    author: dict[str, Any] = _author if isinstance(_author, dict) else {}
    inbox = ZernioInboxRepo(db)
    await inbox.upsert_comment(
        account_id=account_id,
        comment_id=comment_id,
        post_id=_s(comment.get("postId")),
        platform_post_id=_s(comment.get("platformPostId")),
        platform=_s(comment.get("platform")),
        text=_s(comment.get("text")),
        author_id=_s(author.get("id")),
        author_name=_s(author.get("name")),
        author_username=_s(author.get("username")),
        is_reply=bool(comment.get("isReply")),
        parent_id=_s(comment.get("parentCommentId")),
        created_at=_s(comment.get("createdAt")),
    )
    author_key = _s(author.get("id"))
    if author_key:
        await inbox.upsert_contact(
            account_id=account_id,
            contact_key=author_key,
            platform=_s(comment.get("platform")),
            name=_s(author.get("name")),
            username=_s(author.get("username")),
            tag="comment-lead",
        )


async def _process_message(db: Database, payload: dict[str, Any]) -> None:
    """message.received / message.sent → store the message + bump the
    conversation's last_message_at."""
    message = payload.get("message")
    account = payload.get("account")
    if not isinstance(message, dict) or not isinstance(account, dict):
        return
    account_id = _s(account.get("id"))
    message_id = _s(message.get("id"))
    if not (account_id and message_id):
        return
    conversation_id = _s(message.get("conversationId"))
    sent_at = _s(message.get("sentAt"))
    inbox = ZernioInboxRepo(db)
    await inbox.upsert_message(
        account_id=account_id,
        message_id=message_id,
        conversation_id=conversation_id,
        platform=_s(message.get("platform")),
        direction=_s(message.get("direction")),
        text=_s(message.get("text")),
        sent_at=sent_at,
        is_read=bool(message.get("isRead")),
    )
    # Keep the conversation row fresh (an incoming message may precede
    # any conversation.started we saw).
    if conversation_id:
        await inbox.upsert_conversation(
            account_id=account_id,
            conversation_id=conversation_id,
            platform=_s(message.get("platform")),
            participant_id=None,
            participant_name=None,
            participant_username=None,
            status=None,
            last_message_at=sent_at,
        )


async def _process_conversation(db: Database, payload: dict[str, Any]) -> None:
    """conversation.started → upsert the conversation + seed the
    participant as a potential DM-lead contact."""
    conv = payload.get("conversation")
    account = payload.get("account")
    if not isinstance(conv, dict) or not isinstance(account, dict):
        return
    account_id = _s(account.get("id"))
    conversation_id = _s(conv.get("platformConversationId")) or _s(conv.get("id"))
    if not (account_id and conversation_id):
        return
    inbox = ZernioInboxRepo(db)
    await inbox.upsert_conversation(
        account_id=account_id,
        conversation_id=conversation_id,
        platform=_s(conv.get("platform")),
        participant_id=_s(conv.get("participantId")),
        participant_name=_s(conv.get("participantName")),
        participant_username=_s(conv.get("participantUsername")),
        status=_s(conv.get("status")) or "active",
        last_message_at=_s(payload.get("startedAt")),
    )
    participant_key = _s(conv.get("participantId"))
    if participant_key:
        await inbox.upsert_contact(
            account_id=account_id,
            contact_key=participant_key,
            platform=_s(conv.get("platform")),
            name=_s(conv.get("participantName")),
            username=_s(conv.get("participantUsername")),
            tag="dm-lead",
        )


async def _process_whatsapp_number(
    db: Database, event_type: str, payload: dict[str, Any]
) -> None:
    """whatsapp.number.* → record the latest provisioning status for
    the account (phase 12 seam; surfaced only when FEATURE_WHATSAPP is
    on). status is the bit after 'whatsapp.number.'."""
    from chalybclip.db import ZernioWhatsappNumbersRepo

    status = event_type.rsplit(".", 1)[-1]  # activated / declined / ...
    account = payload.get("account")
    account_id = None
    if isinstance(account, dict):
        account_id = account.get("id") or account.get("accountId")
    if not isinstance(account_id, str):
        # Some number events nest the id under a `number`/`phoneNumber`.
        number = payload.get("number") or payload.get("phoneNumber")
        if isinstance(number, dict):
            account_id = number.get("accountId") or number.get("id")
    if not isinstance(account_id, str) or not account_id:
        return
    detail = payload.get("reason") or payload.get("message")
    await ZernioWhatsappNumbersRepo(db).upsert(
        account_id=account_id,
        status=status,
        detail=detail if isinstance(detail, str) else None,
    )


async def _maybe_notify_community(
    db: Database, *, tenant_id: str, post_id: str, post: dict[str, Any]
) -> None:
    """Announce a freshly published clip to the tenant's Discord/
    Telegram community channel, if "Avisar a mi comunidad" is on.

    Guards: (1) skip if THIS post is itself a community announcement we
    created (loop guard); (2) claim the source post id so an
    at-least-once redelivery announces at most once. Never raises."""
    from chalybclip.db import ZernioCommunityRepo, ZernioPublishesRepo
    from chalybclip.settings import get_settings

    from .client import ZernioClient, ZernioError
    from .community import build_notification_payload

    try:
        community = ZernioCommunityRepo(db)
        # Loop guard: a notification's own post.published must not announce.
        if await community.is_notification_post(post_id):
            return
        settings_row = await community.get_settings(tenant_id)
        if not settings_row or not settings_row.get("enabled"):
            return
        discord_id = settings_row.get("discord_account_id")
        telegram_id = settings_row.get("telegram_account_id")
        if not (discord_id or telegram_id):
            return
        # Once-only claim (at-least-once redelivery → announce once).
        if not await community.claim_notification(
            source_post_id=post_id, tenant_id=tenant_id,
        ):
            return

        settings = get_settings()
        if not settings.zernio_api_key:
            return
        tenant = await TenantsRepo(db).get(tenant_id)
        profile_id = tenant.zernio_profile_id if tenant else None
        if not profile_id:
            return

        # Thumbnail: the local publish row's clip thumbnail, if any.
        thumbnail_url = None
        pub = await ZernioPublishesRepo(db).get_by_post_id(post_id)
        if pub is not None:
            base = (settings.public_url or "").rstrip("/")
            if base and pub.clip_id:
                thumbnail_url = f"{base}/dashboard/clips/{pub.clip_id}/thumbnail"

        platforms, psd, _text = build_notification_payload(
            post,
            discord_account_id=discord_id,
            telegram_account_id=telegram_id,
            brand_name=settings_row.get("brand_name"),
            brand_avatar_url=settings_row.get("brand_avatar_url"),
            thumbnail_url=thumbnail_url,
        )
        if not platforms:
            return
        client = ZernioClient(
            api_key=settings.zernio_api_key, base_url=settings.zernio_base_url,
        )
        try:
            result = await client.create_community_post(
                profile_id=profile_id,
                content=_text,
                platforms=platforms,
                platform_specific_data=psd or None,
            )
        except ZernioError as e:
            _log.warning(
                "zernio.community.notify_failed tenant=%s post=%s err=%s",
                tenant_id, post_id, e,
            )
            return
        await community.set_notification_post(
            source_post_id=post_id, notification_post_id=result.post_id,
        )
        _log.info(
            "zernio.community.notified tenant=%s source=%s notif=%s",
            tenant_id, post_id, result.post_id,
        )
    except Exception as e:  # best-effort, never break event processing
        _log.warning(
            "zernio.community.notify_crashed tenant=%s post=%s err=%s",
            tenant_id, post_id, e,
        )


async def _do_auto_retry(db: Database, post_id: str, delay_s: float) -> None:
    """Sleep `delay_s`, then fire ONE retry. Records the outcome on the
    ledger. Never raises (background task)."""
    import asyncio

    from chalybclip.settings import get_settings

    from .client import ZernioClient, ZernioError

    try:
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        settings = get_settings()
        if not settings.zernio_api_key:
            return
        client = ZernioClient(
            api_key=settings.zernio_api_key, base_url=settings.zernio_base_url,
        )
        try:
            await client.retry_post(post_id)
            await ZernioAutoRetriesRepo(db).set_outcome(post_id, outcome="ok")
            _log.info("zernio.auto_retry.ok post_id=%s", post_id)
        except ZernioError as e:
            await ZernioAutoRetriesRepo(db).set_outcome(post_id, outcome="failed")
            _log.info("zernio.auto_retry.failed post_id=%s err=%s", post_id, e)
    except Exception as e:
        _log.warning("zernio.auto_retry.crashed post_id=%s err=%s", post_id, e)


async def _maybe_auto_retry(
    db: Database, *, post_id: str, tenant_id: str | None,
) -> None:
    """Claim + schedule the once-only auto-retry for a transient
    failure. The claim is synchronous (so a redelivery loses the race);
    the retry itself is fire-and-forget after the configured delay.

    With delay 0 (tests / disabled) the retry runs inline so its effect
    is observable; disabled entirely when the delay knob is negative."""
    import asyncio

    from chalybclip.settings import get_settings

    delay_s = get_settings().hub_auto_retry_delay_s
    if delay_s < 0:
        return  # auto-retry disabled
    claimed = await ZernioAutoRetriesRepo(db).claim(post_id, tenant_id=tenant_id)
    if not claimed:
        return  # already auto-retried this post — once only
    if delay_s == 0:
        # Inline: deterministic for tests and for "retry immediately".
        await _do_auto_retry(db, post_id, 0.0)
        return
    task = asyncio.create_task(_do_auto_retry(db, post_id, delay_s))
    _auto_retry_tasks.add(task)
    task.add_done_callback(_auto_retry_tasks.discard)


async def process_pending(db: Database, *, limit: int = 100) -> int:
    """Sweep events whose background task died before mark_processed.

    Returns the number of events processed. Idempotent — safe to run
    on a schedule or at startup."""
    rows = await ZernioEventsRepo(db).list_unprocessed(limit=limit)
    for row in rows:
        await process_zernio_event(db, row.event_id)
    return len(rows)


__all__ = [
    "POST_EVENT_STATUS",
    "SUBSCRIBED_EVENTS",
    "extract_profile_id",
    "process_pending",
    "process_zernio_event",
]
