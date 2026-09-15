"""Convenience helpers for opening the DB + binding a tenant in one go.

The orchestrator and every CLI handler that touches the DB go through
`db_session(...)` so we have a single place where:
    * the SQLite file is opened
    * migrations are applied
    * the tenant is bound to the contextvar
    * the connection is closed on exit (even on error)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from chalybclip.settings import get_settings, resolve_db_target
from chalybclip.tenancy import bound_tenant

from .connection import Database
from .migrations import apply_migrations


@asynccontextmanager
async def db_session(
    *,
    tenant_id: str,
    db_path: str | Path | None = None,
    apply_migrations_on_open: bool = True,
) -> AsyncIterator[Database]:
    """Open the DB, optionally apply migrations, bind the tenant, yield.

    Closes the connection on exit. Works under both the orchestrator and
    individual CLI handlers; both bind the same tenant for the duration.
    """
    # An explicit `db_path` arg always wins (CLI / tests pass a file path).
    # Otherwise resolve via settings, which prefers a Postgres DSN when
    # DATABASE_URL is set. Don't wrap the resolved value in Path() — a DSN
    # string must reach Database() intact for scheme dispatch.
    if db_path is not None:
        target: str | Path = db_path
    else:
        target = resolve_db_target(get_settings())
    db = Database(target)
    try:
        if apply_migrations_on_open:
            await apply_migrations(db)
        with bound_tenant(tenant_id):
            yield db
    finally:
        await db.close()
