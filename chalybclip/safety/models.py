"""Pydantic types for the publishing safe trap.

A `SafetyPolicy` is the per-tenant (per-brand-kit) bundle of per-platform
posting rules. `evaluate_post_window` scores a proposed post against the
relevant `PlatformSafetyRule` and returns a `SafetyVerdict` — advisory by
default, or a hard gate when the brand kit opts into auto-scheduling.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# safe     — clear to post now
# caution  — allowed, but cadence is tight; surfaced as a warning chip
# blocked  — would trip a platform spam/ghost heuristic; defer to recommended_at
Risk = Literal["safe", "caution", "blocked"]


class PlatformSafetyRule(BaseModel):
    """One platform's anti-shadowban posting envelope.

    All times are minutes; quiet hours are hour-of-day integers [0, 23]
    interpreted in the owning `SafetyPolicy.timezone`. A field set to its
    "off" value (0 spacing, 0 cap, None quiet hours) disables that check.
    """

    model_config = ConfigDict(extra="forbid")

    # Minimum minutes between two posts on this platform. 0 == no spacing rule.
    min_spacing_min: int = 0
    # Max posts per local calendar day. 0 == unlimited.
    daily_cap: int = 0
    # Quiet window [start, end) in local hours — posting inside it gets
    # deferred. start > end wraps past midnight (e.g. 23 -> 6). Either None
    # disables the quiet-hours check.
    quiet_hours_start: int | None = None
    quiet_hours_end: int | None = None
    # Deterministic +0..jitter_min minutes nudged onto a computed safe slot
    # so auto-scheduled posts don't fire on a robotic round-number cadence.
    jitter_min: int = 0


class SafetyPolicy(BaseModel):
    """A timezone + the per-platform rules that apply within it.

    `rules` holds fully-resolved `PlatformSafetyRule`s (brand-kit overrides
    already overlaid on the built-in defaults — see `policy.policy_for_kit`).
    `merged()` falls back to the built-in default for any platform not in
    `rules`.
    """

    model_config = ConfigDict(extra="forbid")

    timezone: str = "UTC"
    rules: dict[str, PlatformSafetyRule] = Field(default_factory=dict)

    def merged(self, platform: str) -> PlatformSafetyRule:
        from .policy import default_rule_for

        return self.rules.get(platform) or default_rule_for(platform)


class SafetyVerdict(BaseModel):
    """Outcome of scoring one proposed post against the policy."""

    model_config = ConfigDict(extra="forbid")

    platform: str
    allowed: bool
    risk: Risk
    reason: str
    requested_at: str
    recommended_at: str
