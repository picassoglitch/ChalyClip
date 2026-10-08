"""Repos for the consumption contract: usage outbox, admitted jobs, storage.

Kept out of `repos.py` (already 3k+ lines) because these tables are
billing plumbing, not tenant-scoped domain data: the outbox drain and the
boost entrypoint run with no tenant bound and work across tenants, the
same way `TenantsRepo` does. Every method that reads one tenant's data
still takes `tenant_id` explicitly and filters on it first.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from typing import Any

from .connection import Database


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@dataclass(frozen=True)
class OutboxRow:
    id: str
    tenant_id: str
    endpoint: str
    payload: dict[str, Any]
    status: str
    attempts: int
    next_attempt_at: str
    last_error: str | None
    created_at: str


def _outbox_row(row: Any) -> OutboxRow:
    return OutboxRow(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        endpoint=str(row["endpoint"]),
        payload=json.loads(row["payload_json"] or "{}"),
        status=str(row["status"]),
        attempts=int(row["attempts"] or 0),
        next_attempt_at=str(row["next_attempt_at"]),
        last_error=row["last_error"],
        created_at=str(row["created_at"]),
    )


class UsageOutboxRepo:
    """Durable queue of usage events + reservation settles bound for the hub."""

    def __init__(self, db: Database):
        self._db = db

    async def add(
        self,
        *,
        row_id: str,
        tenant_id: str,
        payload: dict[str, Any],
        endpoint: str = "usage",
    ) -> bool:
        """Insert one pending row. Idempotent on `row_id` (the event's
        source_id) — returns False when it was already queued."""
        now = _now()
        conn = await self._db.connect()
        cur = await conn.execute(
            "INSERT INTO usage_outbox (id, tenant_id, endpoint, payload_json, "
            "status, attempts, next_attempt_at, created_at) "
            "VALUES (?, ?, ?, ?, 'pending', 0, ?, ?) "
            "ON CONFLICT (id) DO NOTHING",
            (row_id, tenant_id, endpoint, json.dumps(payload, ensure_ascii=False), now, now),
        )
        await conn.commit()
        return bool(getattr(cur, "rowcount", 1))

    async def due(self, *, now_iso: str, limit: int) -> list[OutboxRow]:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM usage_outbox WHERE status = 'pending' "
            "AND next_attempt_at <= ? ORDER BY created_at LIMIT ?",
            (now_iso, int(limit)),
        )
        return [_outbox_row(r) for r in await cur.fetchall()]

    async def claim(self, ids: list[str], *, now_iso: str, until_iso: str) -> list[str]:
        """Lease the rows this drain pass will work: push `next_attempt_at`
        out to `until_iso`, but only on rows that are still pending AND due.
        Each row is a single conditional UPDATE (atomic in SQLite and
        Postgres), so when two drains (another web instance, the worker's
        job-end drain) read the same due rows, exactly one of them claims
        each row and only the claimed ids are returned to work on."""
        if not ids:
            return []
        conn = await self._db.connect()
        claimed: list[str] = []
        for i in ids:
            cur = await conn.execute(
                "UPDATE usage_outbox SET next_attempt_at = ? "
                "WHERE id = ? AND status = 'pending' AND next_attempt_at <= ?",
                (until_iso, i, now_iso),
            )
            if getattr(cur, "rowcount", 0) == 1:
                claimed.append(i)
        await conn.commit()
        return claimed

    async def pending_usage_for_reservation(self, reservation_id: str) -> int:
        """How many usage events tied to this reservation are still waiting
        to be delivered. A settle is held back until this is 0, so the hub
        never closes a job's hold before the job's spend has landed."""
        conn = await self._db.connect()
        # Matches the payload as `add` serializes it (json.dumps defaults).
        needle = json.dumps({"reservation_id": reservation_id}, ensure_ascii=False)[1:-1]
        cur = await conn.execute(
            "SELECT COUNT(*) AS n FROM usage_outbox WHERE endpoint = 'usage' "
            "AND status = 'pending' AND payload_json LIKE ?",
            (f"%{needle}%",),
        )
        row = await cur.fetchone()
        return int(row["n"] or 0) if row is not None else 0

    async def defer(self, ids: list[str], *, note: str, next_attempt_at: str) -> None:
        """Push rows out without counting an attempt (nothing was sent)."""
        if not ids:
            return
        conn = await self._db.connect()
        await conn.executemany(
            "UPDATE usage_outbox SET last_error = ?, next_attempt_at = ? "
            "WHERE id = ? AND status = 'pending'",
            [(note[:500], next_attempt_at, i) for i in ids],
        )
        await conn.commit()

    async def mark_sent(self, ids: list[str]) -> None:
        if not ids:
            return
        now = _now()
        conn = await self._db.connect()
        await conn.executemany(
            "UPDATE usage_outbox SET status = 'sent', sent_at = ?, last_error = NULL "
            "WHERE id = ?",
            [(now, i) for i in ids],
        )
        await conn.commit()

    async def mark_retry(self, ids: list[str], *, error: str, next_attempt_at: str) -> None:
        if not ids:
            return
        conn = await self._db.connect()
        await conn.executemany(
            "UPDATE usage_outbox SET attempts = attempts + 1, last_error = ?, "
            "next_attempt_at = ? WHERE id = ? AND status = 'pending'",
            [(error[:500], next_attempt_at, i) for i in ids],
        )
        await conn.commit()

    async def mark_dead(self, ids: list[str], *, error: str) -> None:
        if not ids:
            return
        conn = await self._db.connect()
        await conn.executemany(
            "UPDATE usage_outbox SET status = 'dead', attempts = attempts + 1, "
            "last_error = ? WHERE id = ? AND status = 'pending'",
            [(error[:500], i) for i in ids],
        )
        await conn.commit()

    async def get(self, row_id: str) -> OutboxRow | None:
        conn = await self._db.connect()
        cur = await conn.execute("SELECT * FROM usage_outbox WHERE id = ?", (row_id,))
        row = await cur.fetchone()
        return _outbox_row(row) if row is not None else None

    async def count_by_status(self) -> dict[str, int]:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT status, COUNT(*) AS n FROM usage_outbox GROUP BY status"
        )
        return {str(r["status"]): int(r["n"]) for r in await cur.fetchall()}


@dataclass(frozen=True)
class UsageJobRow:
    id: str
    tenant_id: str
    stream_id: str
    operation: str
    job_class: str
    reservation_id: str | None
    lane: str
    boost: bool | None
    upload_mb: float
    source_minutes: float
    status: str
    kickoff: dict[str, Any] | None


def _job_row(row: Any) -> UsageJobRow:
    boost = row["boost"]
    raw_kickoff = row["kickoff_json"]
    return UsageJobRow(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        stream_id=str(row["stream_id"]),
        operation=str(row["operation"]),
        job_class=str(row["job_class"]),
        reservation_id=row["reservation_id"],
        lane=str(row["lane"] or "standard"),
        boost=None if boost is None else bool(boost),
        upload_mb=float(row["upload_mb"] or 0),
        source_minutes=float(row["source_minutes"] or 0),
        status=str(row["status"]),
        kickoff=json.loads(raw_kickoff) if raw_kickoff else None,
    )


class UsageJobsRepo:
    """One row per admitted run: the external_job_id → reservation mapping,
    the numbers it was admitted with (so the post-probe re-admit can resend
    them), and — for boost runs — the serialized kickoff."""

    def __init__(self, db: Database):
        self._db = db

    async def upsert(
        self,
        *,
        job_id: str,
        tenant_id: str,
        stream_id: str,
        operation: str,
        job_class: str,
        reservation_id: str | None,
        lane: str,
        boost: bool | None,
        upload_mb: float,
        source_minutes: float,
        status: str = "admitted",
    ) -> None:
        now = _now()
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO usage_jobs (id, tenant_id, stream_id, operation, job_class, "
            "reservation_id, lane, boost, upload_mb, source_minutes, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET stream_id = excluded.stream_id, "
            "reservation_id = excluded.reservation_id, lane = excluded.lane, "
            "boost = excluded.boost, upload_mb = excluded.upload_mb, "
            "source_minutes = excluded.source_minutes, status = excluded.status, "
            "updated_at = excluded.updated_at",
            (
                job_id, tenant_id, stream_id, operation, job_class, reservation_id,
                lane, None if boost is None else int(boost), float(upload_mb),
                float(source_minutes), status, now, now,
            ),
        )
        await conn.commit()

    async def get(self, job_id: str) -> UsageJobRow | None:
        conn = await self._db.connect()
        cur = await conn.execute("SELECT * FROM usage_jobs WHERE id = ?", (job_id,))
        row = await cur.fetchone()
        return _job_row(row) if row is not None else None

    async def set_kickoff(self, job_id: str, kickoff: dict[str, Any]) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE usage_jobs SET kickoff_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(kickoff, ensure_ascii=False), _now(), job_id),
        )
        await conn.commit()

    async def set_stream(self, job_id: str, stream_id: str) -> None:
        """Bind the stream id once ingest minted it (admission ran first)."""
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE usage_jobs SET stream_id = ?, updated_at = ? WHERE id = ?",
            (stream_id, _now(), job_id),
        )
        await conn.commit()

    async def set_status(self, job_id: str, status: str) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE usage_jobs SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now(), job_id),
        )
        await conn.commit()


async def tenant_storage_bytes(db: Database, tenant_id: str) -> int:
    """What a tenant holds on this engine: the per-stream sizes measured at
    the end of each successful run. Cheap (one indexed SUM) — admission
    calls it on every upload/kickoff."""
    conn = await db.connect()
    cur = await conn.execute(
        "SELECT COALESCE(SUM(storage_bytes), 0) AS n FROM streams WHERE tenant_id = ?",
        (tenant_id,),
    )
    row = await cur.fetchone()
    return int(row["n"] or 0) if row is not None else 0


async def set_stream_storage_bytes(
    db: Database, *, tenant_id: str, stream_id: str, storage_bytes: int
) -> None:
    conn = await db.connect()
    await conn.execute(
        "UPDATE streams SET storage_bytes = ? WHERE id = ? AND tenant_id = ?",
        (max(0, int(storage_bytes)), stream_id, tenant_id),
    )
    await conn.commit()
