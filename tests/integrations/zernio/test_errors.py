"""Per-platform error classification + Spanish hints (phase 6)."""

from __future__ import annotations

import pytest

from chalybclip.integrations.zernio.errors import (
    classify_category,
    is_transient,
    post_is_auto_retryable,
    spanish_hint,
    summarize_failed_platforms,
)


@pytest.mark.parametrize(
    "category,transient",
    [
        ("platform_error", True),
        ("system_error", True),
        ("user_abuse", True),
        ("auth_expired", False),
        ("user_content", False),
        ("platform_rejected", False),
        ("account_issue", False),
        ("unknown", False),
    ],
)
def test_transient_classification(category: str, transient: bool) -> None:
    assert is_transient(category) is transient


def test_hint_is_spanish_and_action_oriented() -> None:
    assert "reconé" in spanish_hint("auth_expired").lower()
    assert "reintenta" in spanish_hint("platform_error").lower()
    # Every category has a non-empty hint.
    for cat in (
        "auth_expired", "user_content", "user_abuse", "account_issue",
        "platform_rejected", "platform_error", "system_error", "unknown",
    ):
        assert spanish_hint(cat).strip()


def test_message_fallback_when_category_unknown() -> None:
    # No usable category, but the message names the cause.
    assert classify_category("unknown", "Access token expired") == "auth_expired"
    assert classify_category(None, "caption too long") == "user_content"
    assert classify_category("", "Rate limit exceeded") == "user_abuse"
    # Nothing to go on → unknown.
    assert classify_category(None, "weird") == "unknown"


def test_explicit_category_wins_over_message() -> None:
    # A real category is trusted even if the message says otherwise.
    assert classify_category("platform_rejected", "token expired") == "platform_rejected"


def test_summarize_only_failed_platforms() -> None:
    post = {
        "platforms": [
            {"platform": "tiktok", "status": "published"},
            {"platform": "youtube", "status": "failed",
             "errorCategory": "auth_expired",
             "errorMessage": "token expired"},
            "not-a-dict",
        ]
    }
    rows = summarize_failed_platforms(post)
    assert len(rows) == 1
    assert rows[0]["platform"] == "youtube"
    assert rows[0]["category"] == "auth_expired"
    assert rows[0]["transient"] is False
    assert "reconé" in rows[0]["hint"].lower()


def test_auto_retryable_requires_all_failed_transient() -> None:
    # All failed platforms transient → auto-retry once.
    assert post_is_auto_retryable(
        {"platforms": [
            {"platform": "tiktok", "status": "failed",
             "errorCategory": "platform_error"},
        ]}
    )
    # A mix with a human-action failure → do NOT auto-retry (it'd just
    # fail again on the token-expired target).
    assert not post_is_auto_retryable(
        {"platforms": [
            {"platform": "tiktok", "status": "failed",
             "errorCategory": "platform_error"},
            {"platform": "youtube", "status": "failed",
             "errorCategory": "auth_expired"},
        ]}
    )
    # No failures at all → not retryable.
    assert not post_is_auto_retryable(
        {"platforms": [{"platform": "tiktok", "status": "published"}]}
    )
