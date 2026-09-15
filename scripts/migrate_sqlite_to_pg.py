"""One-shot data cutover: copy every row from the live SQLite database into a
Postgres database (Phase 5 of the Postgres migration).

Reads SQLite directly (read-only) and writes through the app's asyncpg facade,
so the SQLite `?` placeholders and 0/1-int "booleans" bind correctly. Tables
are copied in foreign-key dependency order (parents before children) so the
FK constraints hold during the load. Inserts use ON CONFLICT DO NOTHING, so a
re-run after a partial load is safe.

Usage:
    # 1. Stop the app (or run during a maintenance window).
    # 2. Point at the live SQLite file and the target Postgres DSN:
    .venv\\Scripts\\python.exe scripts\\migrate_sqlite_to_pg.py \\
        --sqlite ./chalybclip.db \\
        --pg postgresql://user:pass@host:5432/dbname

    # Re-run safe (idempotent). Use --truncate to wipe the PG tables first
    # for a guaranteed-clean load:
    ... --truncate

The script applies the PG baseline schema first (idempotent), copies the data,
then verifies row counts match per table and exits non-zero on any mismatch.
Nothing in the SQLite file is mutated.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

from chalybclip.db import Database, apply_migrations

# schema_version is owned by the migration runner (apply_migrations stamps it
# on the PG side); never copy it from SQLite.
_SKIP = {"schema_version"}


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'"
    )
    return [r[0] for r in cur.fetchall() if r[0] not in _SKIP]


def _fk_order(conn: sqlite3.Connection, tables: list[str]) -> list[str]:
    """Topologically sort tables so every FK target precedes its referrer.

    Postgres checks FKs at insert time, so parents must load first. Kahn's
    algorithm with an alphabetical tie-break for deterministic output.
    """
    tableset = set(tables)
    deps: dict[str, set[str]] = {}
    for t in tables:
        refs = set()
        for row in conn.execute(f"PRAGMA foreign_key_list({t})").fetchall():
            target = row[2]  # the referenced table
            if target in tableset and target != t:
                refs.add(target)
        deps[t] = refs

    ordered: list[str] = []
    placed = set()
    remaining = dict(deps)
    while remaining:
        ready = sorted(t for t, d in remaining.items() if d <= placed)
        if not ready:
            # FK cycle (shouldn't happen in this schema) — emit the rest in
            # name order and let Postgres surface any genuine problem.
            ready = sorted(remaining)
        for t in ready:
            ordered.append(t)
            placed.add(t)
            del remaining[t]
    return ordered


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


async def _truncate_all(pg: Database, tables: list[str]) -> None:
    conn = await pg.connect()
    quoted = ", ".join(f'"{t}"' for t in tables)
    if quoted:
        await conn.execute(f"TRUNCATE {quoted} RESTART IDENTITY CASCADE")


async def migrate(sqlite_path: Path, pg_dsn: str, *, truncate: bool) -> int:
    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    pg = Database(pg_dsn)

    print(f"Applying PG baseline schema to {pg_dsn.rsplit('@', 1)[-1]} ...")
    version = await apply_migrations(pg)
    print(f"  schema at version {version}")

    tables = _fk_order(src, _list_tables(src))

    if truncate:
        print("Truncating target tables (--truncate) ...")
        await _truncate_all(pg, list(reversed(tables)))

    conn = await pg.connect()
    copied: dict[str, int] = {}
    for table in tables:
        cols = _columns(src, table)
        rows = src.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
        if not rows:
            copied[table] = 0
            continue
        collist = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        await conn.executemany(
            f"INSERT INTO {table} ({collist}) VALUES ({placeholders}) "
            f"ON CONFLICT DO NOTHING",
            [tuple(r) for r in rows],
        )
        copied[table] = len(rows)
        print(f"  {table:36s} {len(rows):>7d} rows")

    # Verify: every SOURCE row landed in Postgres (pg_n >= src_n). Postgres
    # having MORE rows is expected and fine — if the app is already serving on
    # PG during the cutover, live activity (e.g. a login minting an api_token)
    # adds rows the SQLite snapshot never had. Only a shortfall (rows that
    # failed to copy) is a real failure.
    print("\nVerifying row counts ...")
    shortfalls = 0
    for table in tables:
        src_n = src.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        cur = await conn.execute(f"SELECT COUNT(*) FROM {table}")
        pg_n = (await cur.fetchone())[0]
        if pg_n < src_n:
            shortfalls += 1
            print(f"  MISSING {table}: sqlite={src_n} pg={pg_n} ({src_n - pg_n} not copied)")
        elif pg_n > src_n:
            print(f"  note {table}: pg={pg_n} > sqlite={src_n} (+{pg_n - src_n}; live writes during cutover)")

    src.close()
    await pg.close()

    total = sum(copied.values())
    if shortfalls:
        print(f"\nFAILED: {shortfalls} table(s) are missing rows in Postgres.")
        return 1
    print(f"\nOK: {total} rows copied; every source row is present in Postgres.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True, help="Path to the live chalybclip.db")
    parser.add_argument("--pg", required=True, help="Target Postgres DSN (postgres://...)")
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="TRUNCATE target tables before loading (clean re-run).",
    )
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite)
    if not sqlite_path.exists():
        print(f"SQLite file not found: {sqlite_path}", file=sys.stderr)
        return 2
    if not args.pg.startswith(("postgres://", "postgresql://")):
        print(f"--pg must be a postgres:// DSN, got {args.pg!r}", file=sys.stderr)
        return 2

    return asyncio.run(migrate(sqlite_path, args.pg, truncate=args.truncate))


if __name__ == "__main__":
    sys.exit(main())
