"""Export ChalyClip's SQLite DB + on-disk artifacts to a folder for external
analysis (pandas / Excel / BI tools) or for archival before a schema migration.

Usage:
    .venv\\Scripts\\python.exe scripts\\export_data.py
    .venv\\Scripts\\python.exe scripts\\export_data.py --out C:\\some\\folder
    .venv\\Scripts\\python.exe scripts\\export_data.py --tenant aldo
    .venv\\Scripts\\python.exe scripts\\export_data.py --no-manifests

Output layout under <out>/<ISO_timestamp>/:
    tables/<table>.csv       — every table flat-dumped to CSV (nested fields → JSON strings)
    tables/<table>.json      — same rows but with nested fields preserved
    manifests/<stream>.json  — per-stream pipeline manifest (from out/<stream_id>/manifest.json)
    summary.json             — table row counts + on-disk stream count + db_path + schema_version

This is a read-only operation. Nothing in the live DB or the out/ tree is
mutated. Safe to run while the dashboard is up.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any


def _is_nested(value: Any) -> bool:
    """A nested column = looks like JSON. CSV gets the raw string; JSON gets parsed."""
    if not isinstance(value, str):
        return False
    s = value.strip()
    return (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    )


def _maybe_parse_json(value: Any) -> Any:
    if not _is_nested(value):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [r[0] for r in cur.fetchall()]


def _schema_version(conn: sqlite3.Connection) -> int | None:
    try:
        cur = conn.execute("SELECT MAX(version) FROM schema_version")
        return cur.fetchone()[0]
    except sqlite3.OperationalError:
        return None


def _dump_table(
    conn: sqlite3.Connection,
    table: str,
    out_dir: Path,
    tenant_filter: str | None,
) -> int:
    """Dump one table as both CSV and JSON. Returns row count."""
    where = ""
    params: tuple[Any, ...] = ()
    if tenant_filter:
        # Only filter on tables that actually have a tenant_id column.
        col_check = conn.execute(f"PRAGMA table_info({table})").fetchall()
        cols = {c[1] for c in col_check}
        if "tenant_id" in cols:
            where = "WHERE tenant_id = ?"
            params = (tenant_filter,)

    cur = conn.execute(f"SELECT * FROM {table} {where}", params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{table}.csv"
    json_path = out_dir / f"{table}.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in rows:
            writer.writerow(row)

    json_rows = []
    for row in rows:
        record = {col: _maybe_parse_json(val) for col, val in zip(cols, row, strict=True)}
        json_rows.append(record)
    json_path.write_text(json.dumps(json_rows, indent=2, default=str), encoding="utf-8")
    return len(rows)


def _copy_manifests(out_root: Path, dest: Path) -> int:
    """Copy every <stream_id>/manifest.json under out/ to dest/."""
    if not out_root.exists():
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for manifest in out_root.glob("*/manifest.json"):
        stream_id = manifest.parent.name
        shutil.copy2(manifest, dest / f"{stream_id}.json")
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=os.environ.get("CHALYBCLIP_DB_PATH", "./chalybclip.db"),
        help="Path to chalybclip.db (default: ./chalybclip.db or CHALYBCLIP_DB_PATH).",
    )
    parser.add_argument(
        "--out",
        default="./export",
        help="Base output folder; a timestamped subdir is created inside it.",
    )
    parser.add_argument(
        "--tenant",
        default=None,
        help="Filter rows to a single tenant_id (e.g. 'aldo'). Tables without a "
        "tenant_id column are still dumped in full.",
    )
    parser.add_argument(
        "--no-manifests",
        action="store_true",
        help="Skip copying out/<stream_id>/manifest.json files.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("CHALYBCLIP_DEFAULT_OUTPUT_DIR", "./out"),
        help="Path to the on-disk stream artifact dir (where manifest.json files live).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found at {db_path}", file=sys.stderr)
        return 2

    timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out) / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    tables_dir = out_root / "tables"
    manifests_dir = out_root / "manifests"

    print(f"Exporting from {db_path}")
    print(f"           -> {out_root}")
    if args.tenant:
        print(f"           filtered to tenant_id={args.tenant!r}")

    conn = sqlite3.connect(db_path)
    try:
        schema_v = _schema_version(conn)
        tables = _list_tables(conn)
        counts: dict[str, int] = {}
        for table in tables:
            counts[table] = _dump_table(conn, table, tables_dir, args.tenant)
            print(f"  {table:32s} {counts[table]:>6d} rows")
    finally:
        conn.close()

    manifest_count = 0
    if not args.no_manifests:
        manifest_count = _copy_manifests(Path(args.out_dir), manifests_dir)
        print(f"  manifests copied                {manifest_count:>6d}")

    summary = {
        "exported_at": timestamp,
        "db_path": str(db_path.resolve()),
        "schema_version": schema_v,
        "tenant_filter": args.tenant,
        "table_row_counts": counts,
        "manifests_copied": manifest_count,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(f"Done. Open {out_root} to inspect.")
    print(f"  Tables (CSV + JSON):  {tables_dir}")
    if manifest_count:
        print(f"  Stream manifests:     {manifests_dir}")
    print(f"  Summary:              {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
