"""Analytics normalization (phase 7) — the no-fake-zeros rule."""

from __future__ import annotations

from chalybclip.integrations.zernio.analytics import (
    normalize_list,
    normalize_metrics,
    normalize_post,
    sum_headline,
)


def test_missing_metric_is_none_present_zero_is_kept() -> None:
    # views present as 0 → real 0; saves absent → None (renders "—").
    m = normalize_metrics({"likes": 342, "views": 0, "comments": 28})
    assert m["likes"] == 342
    assert m["views"] == 0          # real zero, NOT invented
    assert m["comments"] == 28
    assert m["saves"] is None       # absent → unavailable
    assert m["shares"] is None


def test_normalize_metrics_rejects_bool_and_garbage() -> None:
    m = normalize_metrics({"likes": True, "views": "lots", "shares": 5})
    assert m["likes"] is None       # bool is not a real count
    assert m["views"] is None       # non-numeric → None
    assert m["shares"] == 5


def test_normalize_metrics_empty_input() -> None:
    m = normalize_metrics(None)
    assert all(v is None for v in m.values())


def test_normalize_post_flattens_platforms() -> None:
    post = {
        "_id": "p1",
        "content": "hola mundo",
        "publishedAt": "2026-06-01T10:00:00Z",
        "platformPostUrl": "https://x.com/p/1",
        "analytics": {"likes": 10, "views": 0},
        "platforms": [
            {"platform": "tiktok", "accountUsername": "@me",
             "platformPostUrl": "https://tiktok.com/p/1",
             "analytics": {"likes": 7, "views": 1000}},
        ],
    }
    row = normalize_post(post)
    assert row["post_id"] == "p1"
    assert row["metrics"]["likes"] == 10
    assert row["metrics"]["views"] == 0
    assert row["metrics"]["saves"] is None
    assert row["per_platform"][0]["platform"] == "tiktok"
    assert row["per_platform"][0]["metrics"]["views"] == 1000
    assert row["per_platform"][0]["url"] == "https://tiktok.com/p/1"


def test_normalize_post_handles_platformanalytics_key() -> None:
    # Single-post response uses `platformAnalytics`, list uses `platforms`.
    post = {
        "postId": "p2",
        "analytics": {"likes": 1},
        "platformAnalytics": [
            {"platform": "youtube", "analytics": {"views": 5}},
        ],
    }
    row = normalize_post(post)
    assert row["post_id"] == "p2"
    assert row["per_platform"][0]["platform"] == "youtube"
    assert row["per_platform"][0]["metrics"]["views"] == 5


def test_normalize_list() -> None:
    body = {
        "overview": {"totalPosts": 2},
        "posts": [
            {"_id": "a", "analytics": {"likes": 1}},
            {"_id": "b", "analytics": {"likes": 2}},
            "skip-non-dict",
        ],
    }
    rows = normalize_list(body)
    assert [r["post_id"] for r in rows] == ["a", "b"]


def test_sum_headline_none_only_when_all_missing() -> None:
    rows = [
        {"metrics": normalize_metrics({"likes": 10, "views": 100})},
        {"metrics": normalize_metrics({"likes": 5})},  # no views
    ]
    totals = sum_headline(rows)
    assert totals["likes"] == 15
    assert totals["views"] == 100   # one row had it → sum of available
    # comments + shares absent everywhere → None (renders "—").
    assert totals["comments"] is None
    assert totals["shares"] is None


def test_sum_headline_empty_rows() -> None:
    totals = sum_headline([])
    assert all(v is None for v in totals.values())
