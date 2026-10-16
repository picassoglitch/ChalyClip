"""Durable usage delivery to Chalyb — the outbox + its drain.

Contract (chalyb docs/engines/consumption-contract.md § Delivery): usage
events must survive restarts and scale-to-zero. Every event is written to
the `usage_outbox` table first (`enqueue_usage`), then drained to
POST {CHALYB_BASE_URL}/api/engines/chalybclip/usage in batches of at most
100 events per user, with backoff. Reservation settles ride the same queue
(`enqueue_settle` → POST /usage/settle) so a crash between "run finished"
and "hub told" can't leave a reservation holding the user's tokens until
its TTL.

Response handling per batch:
  * 2xx → rows `sent`; the returned balance refreshes the nav-chip cache.
  * 408 / 429 / 5xx / network → rows stay `pending`, `next_attempt_at`
    backs off exponentially (30s → 1h cap). The hub dedupes on
    (engine, source_id), so re-sending is always safe.
  * any other 4xx → permanent. A multi-event batch is split and re-sent
    one event at a time so one bad event can't sink its neighbours; a
    single rejected event is marked `dead` and logged at ERROR. Never
    dropped — the row stays for the operator.

Drained: on web-box startup and every `usage_outbox_interval_s` (lifespan
loop), right after each enqueue (a background kick, single-flight per DB),
and at the end of every job (the metering wrapper awaits a bounded drain
so a scale-to-zero worker doesn't sit on its own events).

Events are queued even in a process without CHALYB_BASE_URL (a worker
missing the var, local dev): the outbox lives in the shared DB, and any
process that can reach the hub drains it. Only the drain needs the hub
config; without it a pass is a no-op.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as _dt
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import httpx

from chalybclip.db import Database, TenantsRepo
from chalybclip.db.usage_repos import OutboxRow, UsageOutboxRepo
from chalybclip.settings import get_settings

_log = logging.getLogger("chalybclip.chalyb.outbox")

# Contract limits on POST /usage.
MAX_EVENTS_PER_REQUEST = 100
_MAX_AMOUNT = 10**12
_MAX_COST_USD_MICROS = 10**9

# Per-request HTTP timeout. Drains run off the hot path; this only bounds
# how long a dead hub can pin one drain pass.
_DRAIN_TIMEOUT_S = 15.0
# How many due rows one drain pass works through.
_DRAIN_BATCH_ROWS = 500
# Lease taken on the rows a pass is working (see UsageOutboxRepo.lease).
_LEASE_S = 120.0
# Retry backoff: 30s, 60s, 120s, … capped at 1h.
_BACKOFF_BASE_S = 30.0
_BACKOFF_CAP_S = 3600.0

# The reservation a pipeline run was admitted under. Bound by the metering
# wrapper for the whole run; every usage event enqueued inside it (LLM
# calls, transcription, compute) carries it so the hub can net the run's
# spend against its hold. Context-local, so concurrent runs don't mix.
_current_reservation: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "chalybclip_usage_reservation", default=None
)


def bind_reservation(reservation_id: str | None) -> contextvars.Token[str | None]:
    return _current_reservation.set(reservation_id)


def reset_reservation(token: contextvars.Token[str | None]) -> None:
    _current_reservation.reset(token)


def current_reservation() -> str | None:
    return _current_reservation.get()


_UNSET: Any = object()


async def enqueue_usage(
    db: Database,
    *,
    tenant_id: str,
    kind: str,
    amount: int,
    cost_usd_micros: int,
    source_id: str,
    occurred_at_iso: str,
    provider: str | None = None,
    operation: str | None = None,
    reservation_id: str | None = _UNSET,
    metadata: dict[str, Any] | None = None,
    kick: bool = True,
) -> bool:
    """Queue ONE usage event for delivery. Returns True when a row was
    written (False: nothing consumed, or already queued under this
    source_id). Queued even when this process has no CHALYB_BASE_URL: the
    row lives in the shared DB and whichever process can reach the hub
    (the web box's drain loop) delivers it.

    `cost_usd_micros` is required on every event by the contract — the hub
    bills `ceil(cost/4)` tokens off it. `reservation_id` defaults to the
    run's bound reservation (see `bind_reservation`)."""
    amount = min(_MAX_AMOUNT, max(0, int(amount)))
    cost_usd_micros = min(_MAX_COST_USD_MICROS, max(0, int(cost_usd_micros)))
    if amount <= 0 and cost_usd_micros <= 0:
        return False
    if reservation_id is _UNSET:
        reservation_id = current_reservation()

    event: dict[str, Any] = {
        "kind": kind,
        "amount": amount,
        "cost_usd_micros": cost_usd_micros,
        "source_id": source_id,
        "occurred_at": occurred_at_iso,
    }
    if provider:
        event["provider"] = provider
    if operation:
        event["operation"] = operation
    if reservation_id:
        event["reservation_id"] = reservation_id
    if metadata:
        event["metadata"] = metadata

    added = await UsageOutboxRepo(db).add(
        row_id=source_id, tenant_id=tenant_id, payload=event
    )
    if added and kick:
        kick_drain(db)
    return added


async def enqueue_settle(
    db: Database,
    *,
    tenant_id: str,
    reservation_id: str | None,
    outcome: str,
    kick: bool = True,
) -> bool:
    """Queue the terminal settle of a reservation (succeeded / failed /
    cancelled). One per reservation — the first outcome queued wins, which
    matches the hub's "settling twice is a no-op". Heartbeats are sent
    directly (see admission.heartbeat): a late heartbeat is worthless."""
    if not reservation_id:
        return False
    added = await UsageOutboxRepo(db).add(
        row_id=f"settle_{reservation_id}",
        tenant_id=tenant_id,
        endpoint="settle",
        payload={"reservation_id": reservation_id, "outcome": outcome},
    )
    if added and kick:
        kick_drain(db)
    return added


# ---- drain ----------------------------------------------------------------


@dataclass
class DrainReport:
    sent: int = 0
    retried: int = 0
    dead: int = 0
    errors: list[str] = field(default_factory=list)


def _iso(dt: _dt.datetime) -> str:
    return dt.isoformat()


def _backoff_s(attempts: int) -> float:
    return float(min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2 ** max(0, attempts))))


def _is_transient(status: int) -> bool:
    return status in (408, 429) or status >= 500


async def drain_outbox(
    db: Database,
    *,
    client: httpx.AsyncClient | None = None,
    now: _dt.datetime | None = None,
    max_rows: int = _DRAIN_BATCH_ROWS,
) -> DrainReport:
    """One pass over the due rows. Never raises — a failed pass leaves the
    rows pending for the next one."""
    report = DrainReport()
    settings = get_settings()
    base = settings.chalyb_base_url
    token = settings.chalyb_admin_token
    if not base:
        return report
    if not token:
        # Misconfigured instance: leave the rows queued, but make it visible
        # on the affected tenants' chip / diag page.
        try:
            rows = await UsageOutboxRepo(db).due(
                now_iso=_iso(now or _dt.datetime.now(_dt.UTC)), limit=max_rows
            )
        except Exception:  # noqa: BLE001
            return report
        for tenant_id in {r.tenant_id for r in rows if r.endpoint == "usage"}:
            await _record_status(
                db, tenant_id, ok=False,
                error="CHALYB_ADMIN_TOKEN not configured on this instance",
            )
        return report

    now = now or _dt.datetime.now(_dt.UTC)
    repo = UsageOutboxRepo(db)
    try:
        rows = await repo.due(now_iso=_iso(now), limit=max_rows)
        if not rows:
            return report
        await repo.lease(
            [r.id for r in rows], until_iso=_iso(now + _dt.timedelta(seconds=_LEASE_S))
        )
    except Exception as e:  # noqa: BLE001 — a drain pass is best-effort
        _log.warning("outbox drain: read failed: %s", e)
        report.errors.append(str(e))
        return report

    root = base.rstrip("/") + "/api/engines/chalybclip"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=_DRAIN_TIMEOUT_S)
    try:
        usage_rows = [r for r in rows if r.endpoint == "usage"]
        settle_rows = [r for r in rows if r.endpoint == "settle"]

        by_tenant: dict[str, list[OutboxRow]] = defaultdict(list)
        for r in usage_rows:
            by_tenant[r.tenant_id].append(r)
        for tenant_id, tenant_rows in by_tenant.items():
            await _drain_tenant(
                db, http, root=root, headers=headers, tenant_id=tenant_id,
                rows=tenant_rows, now=now, report=report,
            )
        for r in settle_rows:
            await _drain_settle(
                db, http, root=root, headers=headers, row=r, now=now, report=report,
            )
    finally:
        if own_client:
            await http.aclose()
    if report.sent or report.dead or report.retried:
        _log.info(
            "outbox drain: sent=%d retried=%d dead=%d",
            report.sent, report.retried, report.dead,
        )
    return report


async def _drain_tenant(
    db: Database,
    http: httpx.AsyncClient,
    *,
    root: str,
    headers: dict[str, str],
    tenant_id: str,
    rows: list[OutboxRow],
    now: _dt.datetime,
    report: DrainReport,
) -> None:
    repo = UsageOutboxRepo(db)
    try:
        tenant = await TenantsRepo(db).get(tenant_id)
    except Exception as e:  # noqa: BLE001
        await _retry(repo, rows, error=f"tenant lookup failed: {e}", now=now, report=report)
        return
    if tenant is None or not tenant.external_user_id:
        # Can never be delivered as-is (no Chalyb user to bill). Dead, not
        # dropped: an operator can re-link the tenant and re-queue.
        reason = "tenant not found" if tenant is None else "tenant has no external_user_id"
        _log.warning(
            "outbox: %d event(s) undeliverable (%s) · tenant=%s", len(rows), reason, tenant_id
        )
        await repo.mark_dead([r.id for r in rows], error=reason)
        report.dead += len(rows)
        return

    for i in range(0, len(rows), MAX_EVENTS_PER_REQUEST):
        await _post_events(
            db, http, url=f"{root}/usage", headers=headers, tenant_id=tenant_id,
            external_user_id=tenant.external_user_id,
            rows=rows[i : i + MAX_EVENTS_PER_REQUEST], now=now, report=report,
        )


async def _post_events(
    db: Database,
    http: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    tenant_id: str,
    external_user_id: str,
    rows: list[OutboxRow],
    now: _dt.datetime,
    report: DrainReport,
) -> None:
    repo = UsageOutboxRepo(db)
    body = {"external_user_id": external_user_id, "events": [r.payload for r in rows]}
    try:
        resp = await http.post(url, json=body, headers=headers)
    except httpx.TimeoutException:
        await _retry(repo, rows, error="timeout contacting Chalyb", now=now, report=report)
        await _record_status(db, tenant_id, ok=False, error="timeout contacting Chalyb")
        return
    except Exception as e:  # noqa: BLE001 — network / TLS / DNS
        err = f"network error: {type(e).__name__}"
        await _retry(repo, rows, error=err, now=now, report=report)
        await _record_status(db, tenant_id, ok=False, error=err)
        return

    if resp.status_code < 300:
        await repo.mark_sent([r.id for r in rows])
        report.sent += len(rows)
        await _store_balance(db, tenant_id, resp)
        await _record_status(db, tenant_id, ok=True)
        return

    excerpt = (resp.text or "").strip().replace("\n", " ")[:200]
    if _is_transient(resp.status_code):
        err = f"Chalyb HTTP {resp.status_code}" + (f" · {excerpt}" if excerpt else "")
        await _retry(repo, rows, error=err, now=now, report=report)
        await _record_status(db, tenant_id, ok=False, error=err)
        return

    # Permanent rejection. Isolate the offender(s) before declaring death.
    if len(rows) > 1:
        for r in rows:
            await _post_events(
                db, http, url=url, headers=headers, tenant_id=tenant_id,
                external_user_id=external_user_id, rows=[r], now=now, report=report,
            )
        return
    err = f"Chalyb rejected the event: HTTP {resp.status_code}" + (
        f" · body: {excerpt}" if excerpt else ""
    )
    if resp.status_code in (401, 403):
        err += " (check CHALYB_ADMIN_TOKEN)"
    _log.error(
        "outbox: usage event DEAD · tenant=%s source=%s status=%d body=%s",
        tenant_id, rows[0].id, resp.status_code, excerpt,
    )
    await repo.mark_dead([rows[0].id], error=err)
    report.dead += 1
    await _record_status(db, tenant_id, ok=False, error=err)


async def _drain_settle(
    db: Database,
    http: httpx.AsyncClient,
    *,
    root: str,
    headers: dict[str, str],
    row: OutboxRow,
    now: _dt.datetime,
    report: DrainReport,
) -> None:
    repo = UsageOutboxRepo(db)
    try:
        resp = await http.post(f"{root}/usage/settle", json=row.payload, headers=headers)
    except Exception as e:  # noqa: BLE001
        await _retry(repo, [row], error=f"network error: {type(e).__name__}", now=now, report=report)
        return
    if resp.status_code < 300:
        await repo.mark_sent([row.id])
        report.sent += 1
        return
    if _is_transient(resp.status_code):
        await _retry(repo, [row], error=f"Chalyb HTTP {resp.status_code}", now=now, report=report)
        return
    excerpt = (resp.text or "").strip().replace("\n", " ")[:200]
    _log.error(
        "outbox: settle DEAD · reservation=%s status=%d body=%s",
        row.payload.get("reservation_id"), resp.status_code, excerpt,
    )
    await repo.mark_dead([row.id], error=f"HTTP {resp.status_code}: {excerpt}")
    report.dead += 1


async def _retry(
    repo: UsageOutboxRepo,
    rows: list[OutboxRow],
    *,
    error: str,
    now: _dt.datetime,
    report: DrainReport,
) -> None:
    # Rows in one batch share a backoff keyed on the most-tried row.
    attempts = max((r.attempts for r in rows), default=0)
    when = now + _dt.timedelta(seconds=_backoff_s(attempts))
    await repo.mark_retry([r.id for r in rows], error=error, next_attempt_at=_iso(when))
    report.retried += len(rows)
    _log.warning(
        "outbox: %d row(s) will retry at %s (attempt %d): %s",
        len(rows), _iso(when), attempts + 1, error,
    )


async def _store_balance(db: Database, tenant_id: str, resp: httpx.Response) -> None:
    """Persist the hub's returned balance on the tenant row so the nav chip
    renders without its own call. Best-effort."""
    try:
        balance = resp.json().get("balance")
    except Exception:  # noqa: BLE001
        return
    if not isinstance(balance, dict):
        return
    try:
        await TenantsRepo(db).set_balance_cache(
            tenant_id,
            remaining=int(balance.get("remaining", 0) or 0),
            unlimited=bool(balance.get("unlimited", False)),
            monthly_used=int(balance.get("monthlyUsed", 0) or 0),
            at_iso=_dt.datetime.now(_dt.UTC).isoformat(),
        )
    except Exception:  # noqa: BLE001
        _log.exception("balance cache update failed · tenant=%s", tenant_id)


async def _record_status(
    db: Database, tenant_id: str, *, ok: bool, error: str | None = None
) -> None:
    """Report outcome on the tenant row (drives the chip's warning state +
    the diag page). Observability only — never raises."""
    try:
        await TenantsRepo(db).set_usage_report_status(
            tenant_id, ok=ok, at_iso=_dt.datetime.now(_dt.UTC).isoformat(),
            error=None if ok else error,
        )
    except Exception:  # noqa: BLE001
        _log.warning("usage report status write failed · tenant=%s ok=%s", tenant_id, ok)


# ---- background kick ------------------------------------------------------

# One in-flight kick per Database object; a kick that lands while one is
# running just lets the running pass (or the next loop tick) pick it up.
_KICKS: dict[int, asyncio.Task[None]] = {}


def kick_drain(db: Database) -> None:
    """Start a background drain pass soon, without blocking the caller.
    The task is registered with the Database so its close() waits for it."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    existing = _KICKS.get(id(db))
    if existing is not None and not existing.done():
        return

    async def _run() -> None:
        try:
            await drain_outbox(db)
        except Exception:  # noqa: BLE001
            _log.exception("outbox kick drain failed")

    key = id(db)
    task = loop.create_task(_run())
    _KICKS[key] = task
    task.add_done_callback(lambda _t: _KICKS.pop(key, None))
    track = getattr(db, "track_background_task", None)
    if callable(track):
        track(task)


async def drain_until_idle(db: Database, *, timeout_s: float = 30.0) -> DrainReport:
    """Job-end drain: run passes until nothing due is left (or the budget
    runs out). Used where the process may go away right after — the worker
    container, the boost job. Never raises."""
    total = DrainReport()
    try:
        async with asyncio.timeout(timeout_s):
            # Let an in-flight kick finish first so we don't race its lease.
            existing = _KICKS.get(id(db))
            if existing is not None and not existing.done():
                await asyncio.wait({existing})
            while True:
                rep = await drain_outbox(db)
                total.sent += rep.sent
                total.retried += rep.retried
                total.dead += rep.dead
                if rep.sent == 0:
                    break
    except TimeoutError:
        _log.warning("outbox: job-end drain hit its %.0fs budget", timeout_s)
    except Exception:  # noqa: BLE001
        _log.exception("outbox: job-end drain failed")
    return total
