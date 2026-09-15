"""Typed exception hierarchy for ChalyClip.

Per CLAUDE.md: errors are typed; never catch `Exception` broadly except at
process boundaries. Each step in the pipeline raises one of these.
"""

from __future__ import annotations


class ChalybClipError(Exception):
    """Base class for all ChalyClip errors."""


class IngestError(ChalybClipError):
    """VOD download or audio extraction failed."""


class TranscriptionError(ChalybClipError):
    """Whisper transcription failed."""


class DetectionError(ChalybClipError):
    """Trigger detection failed."""


class ClipError(ChalybClipError):
    """ffmpeg cut / reformat failed."""


class LLMError(ChalybClipError):
    """LLM provider call failed (after retries)."""


class VariantError(ChalybClipError):
    """Variant generation failed (clip not found, bad persona, etc.)."""


class TenancyError(ChalybClipError):
    """Tenancy contract violation: no tenant bound, mismatch, unknown token."""


class QuotaExceeded(ChalybClipError):  # noqa: N818  # name pinned by CLAUDE.md
    """Tenant quota would be exceeded by this call."""


class BudgetExceeded(ChalybClipError):  # noqa: N818
    """Tenant's daily LLM USD budget would be exceeded by this call.

    Raised by the BudgetGovernor (P2 Task 1) before LLMRouter issues a
    request. Higher-up callers catch this, emit `llm.budget_exhausted`,
    and halt the current pipeline run cleanly.
    """


class CooldownActive(ChalybClipError):  # noqa: N818
    """Repeated low-confidence rescore verdicts triggered a cooldown.

    The governor refuses new rescore requests until `cooldown_s` after the
    last refusal. Operators can clear it by lowering the threshold or
    waiting it out.
    """
