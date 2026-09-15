"""Database connection layer.

Dual-backend by design so the SQLite → Postgres migration can land in
phases without a flag-day rewrite:

  * Construct with a filesystem ``Path`` (or a non-URL string) and you get
    the original single aiosqlite connection — every existing test and the
    current production deploy hit this path unchanged.
  * Construct with a ``postgres://`` / ``postgresql://`` DSN and you get an
    asyncpg connection pool fronted by a thin aiosqlite-compatible facade.

Both backends expose the SAME surface the repos already use:
``connect()`` returns an object with ``execute()`` (returning a cursor with
``fetchone()`` / ``fetchall()`` / ``rowcount``), ``executemany()``,
``executescript()``, ``commit()`` and ``rollback()``; plus the
``transaction()`` context manager. That's why no repo or service code
changes in Phase 1 — only this file and settings do.

The Postgres facade translates the repos' ``?`` placeholders to asyncpg's
``$1, $2`` form on the fly. asyncpg runs each statement in its own implicit
transaction (autocommit), so the repos' ``commit()`` calls become no-ops —
which matches the existing write-then-commit pattern. Statement-level
atomicity is preserved (a single ``execute`` / ``executemany`` is one txn);
multi-statement atomicity, where it matters, still flows through
``transaction()`` (currently unused by callers — audited in Phase 3).
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:
    import asyncpg

_log = logging.getLogger("chalybclip.db.connection")

# How long close() waits for registered background writers (see
# `track_background_task`) before tearing the backend down anyway. Sized to
# cover one usage-report HTTP attempt (10s) plus its follow-up DB writes;
# a report riding its 30s/120s retry backoff is deliberately left running —
# after close() detaches the handles, a straggler's connect() lazily opens
# a fresh backend instead of hitting a closed pool.
_CLOSE_DRAIN_TIMEOUT_S = 15.0

# Backend-agnostic constraint-violation types (UNIQUE / FK / CHECK). Catch
# this in handlers/repos instead of a driver-specific exception so a
# duplicate or constraint hit is handled the same on SQLite and Postgres.
# aiosqlite raises sqlite3.IntegrityError; asyncpg raises its
# IntegrityConstraintViolationError family. Guarded so the SQLite-only path
# still imports if asyncpg is somehow absent.
try:
    import asyncpg.exceptions as _pg_exc

    IntegrityViolation: tuple[type[Exception], ...] = (
        sqlite3.IntegrityError,
        _pg_exc.IntegrityConstraintViolationError,
    )
except ImportError:  # pragma: no cover — asyncpg is a declared dependency
    IntegrityViolation = (sqlite3.IntegrityError,)


def _is_postgres_dsn(target: Path | str) -> bool:
    """True when `target` names a Postgres server rather than a SQLite file."""
    if isinstance(target, Path):
        return False
    return target.startswith(("postgres://", "postgresql://"))


# ---------------------------------------------------------------------------
# Postgres compatibility facade
#
# asyncpg has no DB-API cursor: `fetch()` returns all rows at once and
# `execute()` returns a status string ("UPDATE 3"). The repos, written
# against aiosqlite, expect `cur = await conn.execute(...)` followed by
# `await cur.fetchone()` / `cur.rowcount`. These small shims bridge the gap.
# ---------------------------------------------------------------------------

# Trailing integer of an asyncpg command tag ("INSERT 0 5" → 5).
_STATUS_TAIL_RE = re.compile(r"(\d+)\s*$")


def _qmark_to_dollar(sql: str) -> str:
    """Rewrite aiosqlite `?` placeholders to asyncpg `$1, $2, …`.

    Skips `?` characters inside single-quoted string literals and
    double-quoted identifiers so a literal question mark in text is never
    mistaken for a parameter. The repos parameterize everything, so in
    practice there are no literal `?`s — but the scanner is correct either
    way and cheap.
    """
    out: list[str] = []
    n = 0
    i = 0
    length = len(sql)
    while i < length:
        ch = sql[i]
        if ch == "'":
            # Single-quoted literal: copy through, honoring '' escapes.
            out.append(ch)
            i += 1
            while i < length:
                out.append(sql[i])
                if sql[i] == "'":
                    if i + 1 < length and sql[i + 1] == "'":
                        out.append(sql[i + 1])
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if ch == '"':
            # Double-quoted identifier: copy through verbatim.
            out.append(ch)
            i += 1
            while i < length:
                out.append(sql[i])
                if sql[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "?":
            n += 1
            out.append(f"${n}")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _returns_rows(sql: str) -> bool:
    """Heuristic: does this statement yield a result set?

    True for SELECT / WITH … / VALUES, and for any statement with a
    RETURNING clause. asyncpg requires `fetch()` for these and `execute()`
    for everything else.
    """
    head = sql.lstrip().lower()
    if head.startswith(("select", "with", "values")):
        return True
    return " returning " in f" {head} "


def _parse_rowcount(status: str) -> int:
    """Affected-row count from an asyncpg command tag ("UPDATE 3" → 3)."""
    m = _STATUS_TAIL_RE.search(status or "")
    return int(m.group(1)) if m else 0


class _PgCursor:
    """aiosqlite-cursor-shaped view over a buffered asyncpg result."""

    __slots__ = ("_idx", "_rows", "rowcount")

    def __init__(self, rows: list[Any], rowcount: int):
        self._rows = rows
        self._idx = 0
        self.rowcount = rowcount

    async def fetchone(self) -> Any | None:
        if self._idx >= len(self._rows):
            return None
        row = self._rows[self._idx]
        self._idx += 1
        return row

    async def fetchall(self) -> list[Any]:
        rows = self._rows[self._idx :]
        self._idx = len(self._rows)
        return rows


def _encode_args(params: Sequence[Any]) -> tuple[Any, ...]:
    """Adapt aiosqlite-style params to asyncpg's stricter binding.

    SQLite stores booleans as 0/1 integers and our schema types those
    columns as `bigint` (never Postgres `boolean`). asyncpg refuses to bind
    a Python `bool` to an integer column, so coerce bool → int. `bool` is a
    subclass of `int`, hence the explicit isinstance check first.
    """
    return tuple(int(p) if isinstance(p, bool) else p for p in params)


async def _run(conn: asyncpg.Connection, sql: str, params: Sequence[Any]) -> _PgCursor:
    """Translate + dispatch one statement on an asyncpg connection."""
    query = _qmark_to_dollar(sql)
    args = _encode_args(params)
    if _returns_rows(query):
        rows = await conn.fetch(query, *args)
        return _PgCursor(list(rows), len(rows))
    status = await conn.execute(query, *args)
    return _PgCursor([], _parse_rowcount(status))


class _PgConnection:
    """Autocommit facade over an asyncpg pool.

    Each `execute`/`executemany` acquires a pooled connection, runs in its
    own implicit transaction, and releases — which is exactly the repos'
    write-then-commit shape. `commit()`/`rollback()` are no-ops because the
    statement already committed.
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> _PgCursor:
        async with self._pool.acquire() as conn:
            return await _run(conn, sql, params)

    async def executemany(
        self, sql: str, seq_of_params: Sequence[Sequence[Any]],
    ) -> _PgCursor:
        query = _qmark_to_dollar(sql)
        rows = [_encode_args(p) for p in seq_of_params]
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.executemany(query, rows)
        return _PgCursor([], len(rows))

    async def executescript(self, script: str) -> _PgCursor:
        # asyncpg runs multiple ';'-separated statements via the simple query
        # protocol when there are no bind params (true for migration files).
        async with self._pool.acquire() as conn:
            await conn.execute(script)
        return _PgCursor([], 0)

    async def commit(self) -> None:  # autocommit — nothing to flush
        return None

    async def rollback(self) -> None:
        return None


class _PgPinnedConnection:
    """Like `_PgConnection` but pinned to ONE asyncpg connection inside an
    open transaction — used by `Database.transaction()`. Commit/rollback are
    driven by the surrounding context manager, so they're no-ops here."""

    def __init__(self, conn: asyncpg.Connection):
        self._conn = conn

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> _PgCursor:
        return await _run(self._conn, sql, params)

    async def executemany(
        self, sql: str, seq_of_params: Sequence[Sequence[Any]],
    ) -> _PgCursor:
        query = _qmark_to_dollar(sql)
        rows = [_encode_args(p) for p in seq_of_params]
        await self._conn.executemany(query, rows)
        return _PgCursor([], len(rows))

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class Database:
    """A single SQLite connection OR an asyncpg pool, behind one facade.

    The backend is chosen by the constructor argument: a filesystem path →
    SQLite (WAL, single connection); a `postgres://` DSN → asyncpg pool.
    """

    def __init__(self, target: Path | str):
        self._is_pg = _is_postgres_dsn(target)
        # SQLite keeps a Path; Postgres keeps the DSN string.
        self.path: Path = Path(".") if self._is_pg else Path(target)
        self._dsn: str = str(target) if self._is_pg else ""
        self._conn: aiosqlite.Connection | None = None
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()
        # Fire-and-forget writers registered via `track_background_task` —
        # close() drains these before tearing down the backend so a late
        # write (step event, usage report) can't hit a closed pool.
        self._bg_tasks: set[asyncio.Task[Any]] = set()

    @property
    def is_postgres(self) -> bool:
        return self._is_pg

    @property
    def target(self) -> str:
        """The connection target (Postgres DSN or SQLite path) this instance
        opened. Lets a background task reopen the SAME database the request
        used — `Database(other.target)` — instead of re-deriving from settings
        (which differs from the active DB in tests)."""
        return self._dsn if self._is_pg else str(self.path)

    async def connect(self) -> Any:
        """Lazily open the backend. Idempotent.

        Returns the live aiosqlite connection (SQLite) or the `_PgConnection`
        facade (Postgres). Repos call this per method and use the result the
        same way regardless of backend.
        """
        if self._is_pg:
            return _PgConnection(await self._ensure_pool())
        return await self._ensure_sqlite()

    async def _ensure_sqlite(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        async with self._lock:
            if self._conn is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                conn = await aiosqlite.connect(self.path)
                conn.row_factory = aiosqlite.Row
                # Pragmas — safe defaults for a single-writer workload.
                await conn.execute("PRAGMA journal_mode = WAL")
                await conn.execute("PRAGMA foreign_keys = ON")
                await conn.execute("PRAGMA synchronous = NORMAL")
                # WAL removes reader/writer contention but NOT writer/writer:
                # a contending writer hits the held lock and, with the default
                # busy_timeout of 0, raises "database is locked" immediately.
                # 5s lets SQLite wait + retry internally instead of 500ing.
                await conn.execute("PRAGMA busy_timeout = 5000")
                await conn.commit()
                self._conn = conn
        return self._conn

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is None:
                import asyncpg

                # min_size keeps a warm connection; max_size caps concurrent
                # server connections per app instance so a request storm can't
                # exhaust the Postgres connection budget.
                self._pool = await asyncpg.create_pool(
                    self._dsn, min_size=1, max_size=10,
                )
        return self._pool

    def track_background_task(self, task: asyncio.Task[Any]) -> None:
        """Register a fire-and-forget task that writes through this Database.

        close() waits (bounded by `_CLOSE_DRAIN_TIMEOUT_S`) for registered
        tasks before tearing down the backend, so a late write — a pipeline
        step event, a usage-report status row — lands instead of dying with
        "pool is closed". Also holds a strong reference, so registered tasks
        can't be garbage-collected mid-flight.
        """
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def close(self) -> None:
        await self._drain_background_tasks()
        # Detach the handles BEFORE awaiting their close: asyncpg's
        # Pool.close() yields while it waits for acquired connections, and a
        # straggler calling connect() in that window would otherwise be
        # handed the closing pool and die with InterfaceError("pool is
        # closed"). Detached, a straggler lazily opens a fresh backend and
        # its write still lands.
        conn, self._conn = self._conn, None
        pool, self._pool = self._pool, None
        if conn is not None:
            await conn.close()
        if pool is not None:
            await pool.close()

    async def _drain_background_tasks(self) -> None:
        """Bounded wait for registered background writers. Never raises —
        task exceptions stay with the tasks (they're all best-effort)."""
        deadline = asyncio.get_running_loop().time() + _CLOSE_DRAIN_TIMEOUT_S
        while True:
            current = asyncio.current_task()
            pending = {t for t in self._bg_tasks if t is not current and not t.done()}
            if not pending:
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                _log.warning(
                    "db close: %d background write task(s) still pending "
                    "after %.0fs drain — closing anyway (stragglers reopen "
                    "lazily)",
                    len(pending), _CLOSE_DRAIN_TIMEOUT_S,
                )
                return
            await asyncio.wait(pending, timeout=remaining)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Any]:
        """Wrap a block in BEGIN / COMMIT, rolling back on exception.

        SQLite: BEGIN/COMMIT on the shared connection. Postgres: a single
        pooled connection held for the block with an explicit asyncpg
        transaction. Currently unused by callers (see module docstring) but
        kept correct for both backends.
        """
        if self._is_pg:
            pool = await self._ensure_pool()
            async with pool.acquire() as raw:
                tx = raw.transaction()
                await tx.start()
                try:
                    yield _PgPinnedConnection(raw)
                except BaseException:
                    await tx.rollback()
                    raise
                else:
                    await tx.commit()
            return
        conn = await self._ensure_sqlite()
        try:
            yield conn
        except BaseException:
            await conn.rollback()
            raise
        else:
            await conn.commit()
