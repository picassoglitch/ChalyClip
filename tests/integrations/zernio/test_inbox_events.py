"""Inbox webhook ingest: comments + DMs + contact seeding (phase 9)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from chalybclip.db import Database, ZernioEventsRepo, ZernioInboxRepo, apply_migrations
from chalybclip.integrations.zernio.events import process_zernio_event

_ACC = "acct_ig_1"


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(tmp_path / "inbox.db")
    await apply_migrations(d)
    try:
        yield d
    finally:
        await d.close()


async def _deliver(db: Database, payload: dict) -> None:
    await ZernioEventsRepo(db).insert_dedup(
        event_id=payload["id"], type=payload["event"], payload=json.dumps(payload),
    )
    await process_zernio_event(db, payload["id"])


# ---- comments ----


@pytest.mark.asyncio
async def test_comment_received_stores_and_seeds_contact(db: Database) -> None:
    await _deliver(db, {
        "id": "ev1", "event": "comment.received",
        "comment": {
            "id": "c1", "postId": None, "platformPostId": "pp1",
            "platform": "instagram", "text": "Buenísimo el clip",
            "author": {"id": "u1", "username": "fan1", "name": "Fan Uno"},
            "createdAt": "2026-06-05T10:00:00Z", "isReply": False,
            "parentCommentId": None,
        },
        "post": {}, "account": {"id": _ACC, "platform": "instagram", "username": "yo"},
        "timestamp": "2026-06-05T10:01:00Z",
    })
    inbox = ZernioInboxRepo(db)
    comments = await inbox.list_comments([_ACC])
    assert len(comments) == 1
    assert comments[0]["text"] == "Buenísimo el clip"
    assert comments[0]["status"] == "active"
    # Author seeded as a comment-lead contact.
    contacts = await inbox.list_contacts([_ACC])
    assert len(contacts) == 1
    assert contacts[0]["contact_key"] == "u1"
    assert "comment-lead" in contacts[0]["tags"]
    assert "instagram" in contacts[0]["tags"]


@pytest.mark.asyncio
async def test_comment_filter_by_post(db: Database) -> None:
    for cid, pp in [("c1", "pp1"), ("c2", "pp2")]:
        await _deliver(db, {
            "id": "ev_" + cid, "event": "comment.received",
            "comment": {"id": cid, "platformPostId": pp, "platform": "instagram",
                        "text": "hola", "author": {"id": "u" + cid},
                        "isReply": False, "parentCommentId": None,
                        "createdAt": "2026-06-05T10:00:00Z"},
            "account": {"id": _ACC, "platform": "instagram", "username": "yo"},
        })
    one = await ZernioInboxRepo(db).list_comments([_ACC], platform_post_id="pp1")
    assert {c["comment_id"] for c in one} == {"c1"}


# ---- conversations + messages ----


@pytest.mark.asyncio
async def test_conversation_started_seeds_dm_lead(db: Database) -> None:
    await _deliver(db, {
        "id": "ev1", "event": "conversation.started",
        "conversation": {
            "id": "int1", "platform": "instagram", "platformConversationId": "conv1",
            "participantId": "p1", "participantName": "Cliente",
            "participantUsername": "cliente1", "status": "active",
        },
        "account": {"id": _ACC, "platform": "instagram", "username": "yo"},
        "startedAt": "2026-06-05T10:00:00Z", "timestamp": "2026-06-05T10:00:00Z",
    })
    inbox = ZernioInboxRepo(db)
    convs = await inbox.list_conversations([_ACC])
    assert len(convs) == 1
    assert convs[0]["conversation_id"] == "conv1"
    assert convs[0]["participant_name"] == "Cliente"
    contacts = await inbox.list_contacts([_ACC], tag="dm-lead")
    assert {c["contact_key"] for c in contacts} == {"p1"}


@pytest.mark.asyncio
async def test_message_received_stores_and_touches_conversation(db: Database) -> None:
    await _deliver(db, {
        "id": "ev1", "event": "message.received",
        "message": {
            "id": "m1", "conversationId": "conv1", "platform": "instagram",
            "platformMessageId": "pm1", "direction": "incoming",
            "text": "¿Tienes más clips?", "attachments": [],
            "sender": {}, "sentAt": "2026-06-05T10:05:00Z", "isRead": False,
        },
        "conversation": {}, "account": {"id": _ACC, "platform": "instagram", "username": "yo"},
        "timestamp": "2026-06-05T10:05:00Z",
    })
    inbox = ZernioInboxRepo(db)
    msgs = await inbox.list_messages([_ACC], conversation_id="conv1")
    assert len(msgs) == 1
    assert msgs[0]["text"] == "¿Tienes más clips?"
    assert msgs[0]["direction"] == "incoming"
    # Conversation row created from the message (no conversation.started seen).
    convs = await inbox.list_conversations([_ACC])
    assert convs[0]["conversation_id"] == "conv1"
    assert convs[0]["last_message_at"] == "2026-06-05T10:05:00Z"


@pytest.mark.asyncio
async def test_message_sent_recorded_as_outgoing(db: Database) -> None:
    await _deliver(db, {
        "id": "ev1", "event": "message.sent",
        "message": {
            "id": "m2", "conversationId": "conv1", "platform": "instagram",
            "platformMessageId": "pm2", "direction": "outgoing",
            "text": "¡Claro!", "attachments": [], "sender": {},
            "sentAt": "2026-06-05T10:06:00Z", "isRead": True,
        },
        "conversation": {}, "account": {"id": _ACC, "platform": "instagram", "username": "yo"},
        "timestamp": "2026-06-05T10:06:00Z",
    })
    msgs = await ZernioInboxRepo(db).list_messages([_ACC], conversation_id="conv1")
    assert msgs[0]["direction"] == "outgoing"


@pytest.mark.asyncio
async def test_contact_tags_merge_across_comment_and_dm(db: Database) -> None:
    # Same author leaves a comment AND starts a DM → both tags merge.
    await _deliver(db, {
        "id": "ev1", "event": "comment.received",
        "comment": {"id": "c1", "platformPostId": "pp1", "platform": "instagram",
                    "text": "hi", "author": {"id": "shared", "name": "Multi"},
                    "isReply": False, "parentCommentId": None,
                    "createdAt": "2026-06-05T10:00:00Z"},
        "account": {"id": _ACC, "platform": "instagram", "username": "yo"},
    })
    await _deliver(db, {
        "id": "ev2", "event": "conversation.started",
        "conversation": {"id": "i1", "platform": "instagram",
                         "platformConversationId": "conv1", "participantId": "shared",
                         "participantName": "Multi", "status": "active"},
        "account": {"id": _ACC, "platform": "instagram", "username": "yo"},
        "startedAt": "2026-06-05T10:10:00Z",
    })
    contacts = await ZernioInboxRepo(db).list_contacts([_ACC])
    assert len(contacts) == 1
    tags = contacts[0]["tags"]
    assert "comment-lead" in tags and "dm-lead" in tags
