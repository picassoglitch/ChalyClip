"""Outbound usage reporter — thin front over the durable outbox.

History: this module used to POST each usage event fire-and-forget from a
background task (2 retries, then dropped; lost outright on a restart).
Delivery now goes through `outbox.py`: events are written to the
`usage_outbox` table and drained in batches with backoff, so nothing is
lost to a deploy or a scale-to-zero worker.

What's left here are the named entry points other modules and tests call:

  * `report_usage` — enqueue ONE event and drain right away (awaited).
    For callers that want the hub to see the event before they continue
    (the per-run base fee, read back by the post-run balance refresh).
  * `report_llm_usage` — the LLM flavour (kind=llm.tokens).

The hot paths (LLM router, transcription, compute metering) call
`outbox.enqueue_usage` directly, which queues and kicks a background
drain without waiting on the network.

Idempotency: Chalyb's usage_events table has UNIQUE (engine_id, source_id),
and the outbox is keyed on source_id too, so re-runs are no-ops.
"""

from __future__ import annotations

from typing import Any

from chalybclip.db import Database

from .outbox import _UNSET, drain_outbox, enqueue_usage


async def report_usage(
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
) -> None:
    """Queue one usage event and drain the outbox now. Never raises —
    a failed delivery stays queued for the next drain."""
    queued = await enqueue_usage(
        db,
        tenant_id=tenant_id,
        kind=kind,
        amount=amount,
        cost_usd_micros=cost_usd_micros,
        source_id=source_id,
        occurred_at_iso=occurred_at_iso,
        provider=provider,
        operation=operation,
        reservation_id=reservation_id,
        metadata=metadata,
        kick=False,
    )
    if queued:
        await drain_outbox(db)


async def report_llm_usage(
    db: Database,
    *,
    tenant_id: str,
    llm_call_id: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd_micros: int = 0,
    provider: str = "anthropic",
    occurred_at_iso: str,
    operation: str | None = None,
) -> None:
    """LLM wrapper over report_usage — kind=llm.tokens, amount = input +
    output tokens, with the token split in metadata."""
    await report_usage(
        db,
        tenant_id=tenant_id,
        kind="llm.tokens",
        amount=max(0, int(input_tokens) + int(output_tokens)),
        cost_usd_micros=cost_usd_micros,
        source_id=llm_call_id,
        occurred_at_iso=occurred_at_iso,
        provider=provider,
        operation=operation,
        metadata={
            "tokens": {
                "input": int(input_tokens),
                "output": int(output_tokens),
                "cache_read": 0,
                "cache_write": 0,
            }
        },
    )
