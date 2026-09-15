"""_order_publish_history — Publicados feed ordering: upcoming scheduled
posts lead (soonest-to-publish first), published history follows (newest
first), and upcoming posts are never truncated by the published-tail cap."""

from __future__ import annotations

from typing import Any

from chalybclip.api.routers.zernio import _order_publish_history


def _row(
    post_id: str, *, status: str, created_at: str, scheduled_for: str | None = None
) -> dict[str, Any]:
    return {
        "post_id": post_id,
        "status": status,
        "created_at": created_at,
        "scheduled_for": scheduled_for,
    }


def test_upcoming_scheduled_lead_soonest_first() -> None:
    rows = [
        _row("pub_old", status="published", created_at="2026-06-10T00:00:00Z"),
        _row(
            "sched_late", status="scheduled", created_at="2026-06-16T00:00:00Z",
            scheduled_for="2026-06-24T00:00:00Z",
        ),
        _row(
            "sched_soon", status="scheduled", created_at="2026-06-17T00:00:00Z",
            scheduled_for="2026-06-23T00:00:00Z",
        ),
        _row("pub_new", status="published", created_at="2026-06-15T00:00:00Z"),
    ]
    ordered = [h["post_id"] for h in _order_publish_history(rows)]
    # Upcoming first (soonest publish at the very top), then published newest-first.
    assert ordered == ["sched_soon", "sched_late", "pub_new", "pub_old"]


def test_published_tail_capped_but_upcoming_never_dropped() -> None:
    upcoming = [
        _row(
            f"sched_{i}", status="scheduled", created_at="2026-06-16T00:00:00Z",
            scheduled_for=f"2026-06-{20 + i:02d}T00:00:00Z",
        )
        for i in range(5)
    ]
    published = [
        _row(f"pub_{i}", status="published", created_at=f"2026-06-{1 + i:02d}T00:00:00Z")
        for i in range(40)
    ]
    ordered = _order_publish_history(upcoming + published)
    ids = [h["post_id"] for h in ordered]
    # Every upcoming post survives; the published tail is capped at 25.
    assert all(f"sched_{i}" in ids for i in range(5))
    assert sum(1 for h in ordered if h["status"] == "published") == 25
    # All upcoming rows come before any published row.
    first_pub = next(i for i, h in enumerate(ordered) if h["status"] == "published")
    assert all(h["status"] == "scheduled" for h in ordered[:first_pub])


def test_scheduled_without_time_falls_back_to_created_at() -> None:
    # A 'scheduled' row with no scheduled_for has no publish time to lead
    # with — it sorts by created_at alongside the published rows.
    rows = [
        _row("no_time", status="scheduled", created_at="2026-06-01T00:00:00Z"),
        _row("pub", status="published", created_at="2026-06-02T00:00:00Z"),
    ]
    ordered = [h["post_id"] for h in _order_publish_history(rows)]
    assert ordered == ["pub", "no_time"]  # newest-created first; neither leads
