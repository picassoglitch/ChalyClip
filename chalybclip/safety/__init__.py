"""Publishing safe trap — anti-shadowban posting windows.

Per-platform rules (min spacing, daily cap, quiet hours, jitter) produce
an advisory risk score + a recommended safe time, and — when a brand kit
opts in — drive auto-scheduling so the publish flow runs hands-off without
tripping platform spam/ghost heuristics.

Public surface:
  * `PlatformSafetyRule`, `SafetyPolicy`, `SafetyVerdict` — models
  * `PLATFORM_DEFAULTS`, `default_rule_for`, `policy_for_kit` — config
  * `evaluate_post_window` (advisory), `next_safe_slot` (auto-schedule)
"""

from __future__ import annotations

from .models import PlatformSafetyRule, SafetyPolicy, SafetyVerdict
from .policy import PLATFORM_DEFAULTS, default_rule_for, policy_for_kit
from .service import evaluate_post_window, next_safe_slot

__all__ = [
    "PLATFORM_DEFAULTS",
    "PlatformSafetyRule",
    "SafetyPolicy",
    "SafetyVerdict",
    "default_rule_for",
    "evaluate_post_window",
    "next_safe_slot",
    "policy_for_kit",
]
