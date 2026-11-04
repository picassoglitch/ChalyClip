"""Pre-run token cost estimator for the ChalyClip pipeline.

Given a stream that's about to run, estimate the rough token budget the
LLM-consuming steps will use. Returned as a single integer + a short
breakdown by phase so the dashboard can render a "this run will use ~X
tokens" badge on the Run pipeline button.

This is INTENTIONALLY rough. The pipeline's actual LLM cost depends on
factors we can't know ahead of time:

  * How many candidates pass the detection threshold
  * How many of those become cuts (the detector can suggest 40,
    cutting picks the top 8-12)
  * Variant generation runs N variants per cut × M personas — N and
    persona count we know, but the prompt expansion per variant
    depends on transcript length around the clip
  * Vision rescore (when enabled) hits Claude with image attachments —
    much higher per-call cost
  * Retries on validation failures

So we ship a budget estimate using empirical multipliers from
historical runs (see the constants below). The estimate is presented
to the user as an upper-bound — "no más de ~X tokens" — so an actual
run that comes in 30% lower feels good, not deceptive.

If we ever need precision we'd run a dry-pass through the LLM with
small prompts to count tokens, but that itself costs tokens and the
asymmetry doesn't pay off for our current scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import ValidationError

from chalybclip.errors import ChalybClipError

from .config import LLMConfig, get_llm_config
from .prices import TokenRates, lookup_rates

_log = logging.getLogger("chalybclip.llm.estimator")

# Empirical multipliers — measured across ~30 ChalyClip pipeline runs in
# Q1-Q2 2026. These are 80th-percentile values so we overestimate slightly;
# the worst real-world miss is ~15% under the estimate, which means the
# user sees the bar move LESS than they expected → positive surprise.
#
# Each constant captures one phase's token spend, normalized by whatever
# the phase scales with (clip count, persona count, stream minutes).
TOKENS_PER_CANDIDATE_DETECT = 200      # detector LLM scoring per candidate
TOKENS_PER_CLIP_CUT = 400              # cut step: title, summary, etc.
TOKENS_PER_VARIANT = 1_100             # variant generation per (clip × persona)
TOKENS_PER_HOOK_PER_CLIP = 600         # hook writer per cut
TOKENS_FIXED_OVERHEAD = 1_500          # init + config validation + misc

# Heuristics that drive the above counts when we don't have exact info.
DEFAULT_CANDIDATES_PER_MINUTE = 1.5    # detector typically surfaces ~1.5/min
DEFAULT_CLIPS_PER_HOUR = 8             # cut step normally produces 6-10/hour
DEFAULT_VARIANTS_PER_CLIP = 5          # config default n_variants
DEFAULT_HOOKS_PER_CLIP = 3             # hook writer default fanout

# Token T3: the estimate is priced at the routed `standard` model's rates,
# weighted 3:1 toward output because captions/hooks generation dominates spend.
_OUTPUT_WEIGHT = 0.75
# When the config can't say what runs, reserve at the committed premium model:
# over-reserving beats under-, but the priciest family would be ~100x over.
FALLBACK_MODEL = "claude-opus-5-5"


def _blend(rates: TokenRates) -> float:
    return _OUTPUT_WEIGHT * rates.output + (1 - _OUTPUT_WEIGHT) * rates.input


def _fallback_rate() -> float:
    rates = lookup_rates("anthropic", FALLBACK_MODEL)
    if rates is None:  # prices.py ships with it and a test pins it
        raise LookupError(f"no committed price for {FALLBACK_MODEL}")
    return _blend(rates)


def standard_model(config: LLMConfig) -> tuple[str, str] | None:
    """(provider, model) the standard purposes route to: the first standard rule's primary."""
    for rule in config.routing.values():
        if rule.default_quality == "standard" and rule.primary in config.providers:
            return rule.primary, config.model_for(rule.primary, "standard")
    return None


def blended_usd_per_mtok(config: LLMConfig | None = None) -> float:
    """USD per 1M estimated tokens, from the loaded LLM config (a YAML rollback follows)."""
    try:
        cfg = config if config is not None else get_llm_config()
    except (ChalybClipError, ValidationError, OSError):
        _log.error("llm.estimate_config_unreadable fallback=%s", FALLBACK_MODEL, exc_info=True)
        return _fallback_rate()
    routed = standard_model(cfg)
    if routed is None:
        _log.warning("llm.estimate_nothing_routed fallback=%s", FALLBACK_MODEL)
        return _fallback_rate()
    rates, known = cfg.rates_for(*routed)
    if not known:
        _log.error(
            "llm.price_unknown provider=%s model=%s — estimated at the conservative rate; "
            "add it to chalybclip/llm/prices.py",
            *routed,
        )
    return _blend(rates)


@dataclass(frozen=True)
class EstimateLine:
    """One row in the breakdown shown to the user."""

    phase: str
    tokens: int
    note: str


@dataclass(frozen=True)
class PipelineEstimate:
    """Sum + breakdown for one prospective pipeline run.

    `total_tokens` is the LLM token estimate (the breakdown lines).
    Token T3 adds the COST view so the estimate is comparable to the
    real per-stream spend (which is in USD across all providers):
      - llm_cost_usd_micros           — token estimate × blended rate
      - transcription_cost_usd_micros — real, near-deterministic in
                                        duration (cost_micros_for)
      - total_cost_usd_micros         — all-in estimate to compare
                                        against StreamSpend after a run.
    """

    total_tokens: int
    lines: tuple[EstimateLine, ...]
    llm_cost_usd_micros: int = 0
    transcription_cost_usd_micros: int = 0
    total_cost_usd_micros: int = 0

    @property
    def lines_list(self) -> list[EstimateLine]:
        """Jinja2 doesn't love tuples in for-loops; convenience accessor."""
        return list(self.lines)

    @property
    def total_cost_usd(self) -> float:
        return self.total_cost_usd_micros / 1_000_000.0


def estimate_pipeline_run(
    *,
    duration_seconds: float | None,
    persona_count: int = 1,
    n_variants: int | None = None,
    n_hooks_per_clip: int | None = None,
    llm_config: LLMConfig | None = None,
) -> PipelineEstimate:
    """Estimate the LLM token cost for a single pipeline run.

    Args:
      duration_seconds: stream length. Required for sensible numbers; if
        unknown (rare — ingest fills this in), we fall back to a 1-hour
        assumption with a wide multiplier.
      persona_count: how many personas the variant step will run. Pipeline
        defaults to 1, but partner setups might run all 4.
      n_variants: variants per clip per persona. Defaults to the pipeline's
        config default.
      n_hooks_per_clip: hooks per cut. Defaults to config default.

    Returns:
      PipelineEstimate with a flat total + a breakdown the UI can render.
    """
    # Slice O.47 — variants phase suppressed when variants_enabled is
    # False (the new default). Pipeline still creates one stub variant
    # row per clip but it's a 0-token operation. We omit the line
    # entirely so the UI breakdown doesn't show a phantom "Variantes"
    # row at 0 tokens.
    try:
        from chalybclip.settings import get_settings
        _variants_enabled = bool(get_settings().variants_enabled)
    except Exception:  # noqa: BLE001 — defensive; estimator must never 500
        _variants_enabled = False

    duration_min = (duration_seconds or 3600) / 60.0
    duration_hr = duration_min / 60.0

    candidates = max(1, int(duration_min * DEFAULT_CANDIDATES_PER_MINUTE))
    clips = max(1, int(duration_hr * DEFAULT_CLIPS_PER_HOUR))
    variants_per_clip = n_variants or DEFAULT_VARIANTS_PER_CLIP
    hooks_per_clip = n_hooks_per_clip or DEFAULT_HOOKS_PER_CLIP

    detect_tokens = candidates * TOKENS_PER_CANDIDATE_DETECT
    cut_tokens = clips * TOKENS_PER_CLIP_CUT
    variant_tokens = (
        clips * persona_count * variants_per_clip * TOKENS_PER_VARIANT
        if _variants_enabled
        else 0
    )
    hook_tokens = clips * hooks_per_clip * TOKENS_PER_HOOK_PER_CLIP

    lines_list: list[EstimateLine] = [
        EstimateLine(
            phase="Detección",
            tokens=detect_tokens,
            note=f"~{candidates} candidatos × {TOKENS_PER_CANDIDATE_DETECT} tokens",
        ),
        EstimateLine(
            phase="Cortes",
            tokens=cut_tokens,
            note=f"~{clips} clips × {TOKENS_PER_CLIP_CUT} tokens",
        ),
    ]
    if _variants_enabled:
        lines_list.append(
            EstimateLine(
                phase="Variantes",
                tokens=variant_tokens,
                note=(
                    f"{clips} × {persona_count} persona{'s' if persona_count != 1 else ''} × "
                    f"{variants_per_clip} variantes × {TOKENS_PER_VARIANT} tokens"
                ),
            )
        )
    lines_list.append(
        EstimateLine(
            phase="Hooks",
            tokens=hook_tokens,
            note=f"{clips} × {hooks_per_clip} hooks × {TOKENS_PER_HOOK_PER_CLIP} tokens",
        )
    )
    lines_list.append(
        EstimateLine(
            phase="Overhead",
            tokens=TOKENS_FIXED_OVERHEAD,
            note="init + validación + misc",
        )
    )
    lines = tuple(lines_list)

    total = sum(line.tokens for line in lines)

    # Token T3 — cost view. LLM cost from the token estimate at a blended
    # rate; transcription cost is real (near-deterministic in duration).
    # cost_micros = tokens × usd_per_mtok (the /1e6 and ×1e6 cancel).
    llm_cost_um = int(round(total * blended_usd_per_mtok(llm_config)))
    transcription_cost_um = 0
    try:
        from chalybclip.transcribe.providers.assemblyai import cost_micros_for
        transcription_cost_um = cost_micros_for(
            duration_s=(duration_seconds or 3600), speaker_labels=True,
        )
    except Exception:  # noqa: BLE001 — estimator must never 500
        transcription_cost_um = 0

    return PipelineEstimate(
        total_tokens=total,
        lines=lines,
        llm_cost_usd_micros=llm_cost_um,
        transcription_cost_usd_micros=transcription_cost_um,
        total_cost_usd_micros=llm_cost_um + transcription_cost_um,
    )


def format_tokens(n: int) -> str:
    """Pretty-print a token count: 5.2k, 1.4M, etc. Used by templates."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def estimate_runs_per_balance(
    remaining_tokens: int | None,
    *,
    duration_seconds: float | None,
    persona_count: int = 1,
) -> int | None:
    """How many runs of this stream fit in the user's remaining budget.

    Returns None when balance is unlimited or unknown. Useful for the
    "runs you can afford" hint next to the estimate.
    """
    if remaining_tokens is None or remaining_tokens <= 0:
        return None
    est = estimate_pipeline_run(
        duration_seconds=duration_seconds,
        persona_count=persona_count,
    )
    if est.total_tokens <= 0:
        return None
    return max(0, remaining_tokens // est.total_tokens)


__all__ = [
    "EstimateLine",
    "PipelineEstimate",
    "estimate_pipeline_run",
    "estimate_runs_per_balance",
    "format_tokens",
]
