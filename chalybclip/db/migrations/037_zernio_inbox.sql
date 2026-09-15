-- Inbox: comments + DMs + contacts (Hub phase 9).
--
-- Like the calendar (migration 036), inbox webhooks carry the Zernio
-- social-account id but no profileId, so every table is keyed by
-- account_id and resolved to a tenant at READ time (the route matches
-- account_id against the viewing tenant's connected accounts — that
-- match is the isolation boundary). Webhook-first: the event processor
-- writes here; REST is backfill. The UI reads local state.

-- Comments on the streamer's posts (comment.received).
CREATE TABLE IF NOT EXISTS zernio_comments (
    account_id       TEXT NOT NULL,   -- Zernio social account id
    comment_id       TEXT NOT NULL,   -- platform comment id
    post_id          TEXT,            -- internal post id (null for non-Zernio posts)
    platform_post_id TEXT,            -- platform's post id
    platform         TEXT,
    text             TEXT,
    author_id        TEXT,
    author_name      TEXT,
    author_username  TEXT,
    is_reply         INTEGER NOT NULL DEFAULT 0,
    parent_id        TEXT,
    status           TEXT NOT NULL DEFAULT 'active',  -- active | hidden | deleted
    created_at       TEXT,            -- platform createdAt
    received_at      TEXT NOT NULL,
    PRIMARY KEY (account_id, comment_id)
);
CREATE INDEX IF NOT EXISTS idx_comments_account
    ON zernio_comments (account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_comments_post
    ON zernio_comments (platform_post_id);

-- DM conversations (conversation.started + message.* upserts).
CREATE TABLE IF NOT EXISTS zernio_conversations (
    account_id            TEXT NOT NULL,
    conversation_id       TEXT NOT NULL,  -- platform conversation id
    platform              TEXT,
    participant_id        TEXT,
    participant_name      TEXT,
    participant_username  TEXT,
    status                TEXT NOT NULL DEFAULT 'active',  -- active | archived
    last_message_at       TEXT,
    updated_at            TEXT NOT NULL,
    PRIMARY KEY (account_id, conversation_id)
);
CREATE INDEX IF NOT EXISTS idx_conversations_account
    ON zernio_conversations (account_id, last_message_at DESC);

-- DM messages (message.received / message.sent).
CREATE TABLE IF NOT EXISTS zernio_messages (
    account_id       TEXT NOT NULL,
    message_id       TEXT NOT NULL,   -- internal message id
    conversation_id  TEXT,
    platform         TEXT,
    direction        TEXT,            -- incoming | outgoing
    text             TEXT,
    sent_at          TEXT,
    is_read          INTEGER NOT NULL DEFAULT 0,
    received_at      TEXT NOT NULL,
    PRIMARY KEY (account_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON zernio_messages (account_id, conversation_id, sent_at);

-- Potential contacts auto-seeded from comment + DM authors (feeds the
-- phase-10 growth layer). tags is a csv (e.g. "comment-lead,instagram").
CREATE TABLE IF NOT EXISTS zernio_contacts (
    account_id    TEXT NOT NULL,
    contact_key   TEXT NOT NULL,      -- platform author/participant id
    platform      TEXT,
    name          TEXT,
    username      TEXT,
    tags          TEXT,               -- csv: comment-lead | dm-lead | <platform>
    zernio_contact_id TEXT,           -- Zernio CRM contact id when known (phase 10)
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    PRIMARY KEY (account_id, contact_key)
);
CREATE INDEX IF NOT EXISTS idx_contacts_account
    ON zernio_contacts (account_id, last_seen DESC);
