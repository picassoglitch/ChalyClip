"""Vision-LLM rescore — promote real reactions, suppress loud-but-empty moments.

The Phase 1 detectors are good at finding *something happened* (voice phrase,
chat spike, audio spike, scene cut). They can't tell you whether the streamer
actually reacted on camera. That's what this stage adds: take the top-K
heuristic candidates, sample a few frames around each, and ask a vision LLM
which of them are genuinely clip-worthy.

Three guardrails wrap every call:

  * `BudgetGovernor.check_llm_spend` — refuses cleanly when today's tenant
    LLM total has hit the daily ceiling.
  * `BudgetGovernor.acquire_rescore_slot` — caps concurrent rescores per
    tenant from `tenants.rescore_concurrency_cap`.
  * `BudgetGovernor.record_rescore_verdict` + `check_rescore_cooldown` —
    if the last K verdicts are all below `low_confidence_threshold`, we
    refuse new rescores for `cooldown_s`. Prevents an aggressive rescoring
    loop from burning premium tokens against an obviously-low-content VOD.

Output: each candidate's `rescore_score` / `rescore_reason` / `rescore_model`
columns are populated, and the returned list is re-ranked so high-rescore
items lead.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from dataclasses import dataclass

import structlog

from chalybclip.db import CandidatesRepo, Database
from chalybclip.errors import BudgetExceeded, CooldownActive, DetectionError, LLMError
from chalybclip.events import emit
from chalybclip.governance import BudgetGovernor
from chalybclip.ingest import Stream
from chalybclip.llm import (
    FrameStore,
    LLMRouter,
    MemoryFrameStore,
    MultimodalImage,
    RescoreVerdict,
)
from chalybclip.llm.config import Quality
from chalybclip.tenancy import bound_tenant
from chalybclip.vision.frame_sampler import sample_frames

from .models import Candidate

_log = structlog.get_logger(__name__)

_PURPOSE = "vision_rescore"
_DEFAULT_TOP_K = 10
_DEFAULT_N_FRAMES = 5
_DEFAULT_FRAME_SPREAD_S = 1.5  # cover ~750ms either side of the anchor


@dataclass(frozen=True)
class RescoreOutcome:
    """Roll-up returned to callers + the pipeline orchestrator."""

    rescored: int  # number of candidates that received a verdict
    skipped_budget: int  # halted by BudgetExceeded mid-pass
    skipped_cooldown: int  # halted by CooldownActive at the gate
    fatal_errors: int  # unhandled provider errors after retries


_RESCORE_SYSTEM_PROMPT = (
    "You are reviewing short moments from a livestream VOD to decide which "
    "ones are genuinely clip-worthy.\n\n"
    "Score each moment on the question: does this look like a real on-screen "
    "reaction (laugh, shock, anger, surprised face, big gesture) - or is it "
    "just a loud noise / chat spike with the streamer off-camera or expressionless?\n\n"
    "Use the 0.0-1.0 scale:\n"
    "  * 0.00-0.30: weak. No visible reaction; suppress this candidate.\n"
    "  * 0.30-0.65: ambiguous; the heuristic ranking should stand.\n"
    "  * 0.65-1.00: strong reaction visible on camera; promote it.\n\n"
    "When you can read the streamer's face, set face_emotion to one of "
    "{neutral, smile, laugh, shock, anger, sad}. Otherwise set it to null.\n\n"
    "Keep `reason` to one short sentence (under 30 words) - it ends up in "
    "the human reviewer's panel."
)


def _user_prompt_for(candidate: Candidate, n_frames: int) -> str:
    """Compact briefing for the model."""
    evidence = candidate.evidence or {}
    lines = [
        f"Moment at {candidate.timestamp:.1f}s in the source stream.",
        f"Heuristic detector that fired: {candidate.reason!r} "
        f"(score={candidate.score:.3f}).",
        f"You see {n_frames} frames sampled across ~1.5s around that moment.",
    ]
    phrase = evidence.get("phrase")
    if isinstance(phrase, str) and phrase:
        lines.append(f"Voice trigger phrase: {phrase!r}.")
    snippet = evidence.get("transcript_snippet")
    if isinstance(snippet, str) and snippet:
        lines.append(f"Transcript at trigger: {snippet!r}.")
    chat = evidence.get("chat_snippet")
    if isinstance(chat, str) and chat:
        lines.append(f"Chat at trigger: {chat!r}.")
    return "\n".join(lines)


async def rescore_candidates(
    *,
    tenant_id: str,
    stream: Stream,
    candidates: list[Candidate],
    db: Database,
    router: LLMRouter,
    governor: BudgetGovernor | None = None,
    frame_store: FrameStore | None = None,
    top_k: int = _DEFAULT_TOP_K,
    n_frames_per_candidate: int = _DEFAULT_N_FRAMES,
    quality: Quality | None = None,
) -> tuple[list[Candidate], RescoreOutcome]:
    """Run vision rescore on the top-K candidates.

    Returns `(reranked_candidates, outcome)`. The list mirrors the input
    set but reordered: rescored items first (sorted by rescore_score desc),
    remaining items follow in their original heuristic order. Each rescored
    Candidate carries the verdict in `evidence["rescore"]` so downstream
    serialization survives the round-trip; the persistent record lives on
    `candidates.rescore_*` columns via `CandidatesRepo.update_rescore`.
    """
    if tenant_id != stream.tenant_id:
        raise DetectionError(
            f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}"
        )
    if top_k < 1:
        raise DetectionError(f"top_k must be >= 1, got {top_k}")
    if n_frames_per_candidate < 1:
        raise DetectionError(
            f"n_frames_per_candidate must be >= 1, got {n_frames_per_candidate}"
        )
    store: FrameStore = frame_store if frame_store is not None else MemoryFrameStore()

    if not candidates:
        return [], RescoreOutcome(rescored=0, skipped_budget=0, skipped_cooldown=0, fatal_errors=0)

    # Cooldown gate runs once at the top - if the tenant is in cooldown, no
    # rescore happens at all this pass.
    if governor is not None:
        try:
            await governor.check_rescore_cooldown(tenant_id)
        except CooldownActive as e:
            _log.info("rescore_cooldown_active", tenant_id=tenant_id, error=str(e))
            with bound_tenant(tenant_id):
                await emit(db, "vision_rescore.cooldown_active", {"error": str(e)})
            return list(candidates), RescoreOutcome(
                rescored=0, skipped_budget=0, skipped_cooldown=len(candidates), fatal_errors=0
            )

    sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    top = sorted_candidates[:top_k]
    rest = sorted_candidates[top_k:]

    rescored: list[Candidate] = []
    skipped_budget = 0
    fatal = 0
    halted = False

    for candidate in top:
        if halted:
            break
        try:
            verdict = await _rescore_one(
                tenant_id=tenant_id,
                stream=stream,
                candidate=candidate,
                router=router,
                governor=governor,
                frame_store=store,
                n_frames=n_frames_per_candidate,
                quality=quality,
            )
        except BudgetExceeded:
            # Stop the whole pass - the next one would refuse too. Earlier
            # rescored items keep their verdicts.
            skipped_budget = len(top) - len(rescored)
            halted = True
            break
        except LLMError as e:
            _log.warning(
                "rescore_provider_error",
                tenant_id=tenant_id,
                ts=candidate.timestamp,
                error=str(e),
            )
            fatal += 1
            continue

        if governor is not None:
            governor.record_rescore_verdict(tenant_id, verdict.score)

        # Late import - candidate_pk lives in db.adapters, which transitively
        # imports chalybclip.clip -> chalybclip.detect, so we can't import it at
        # module top.
        from chalybclip.db.adapters import candidate_pk

        with bound_tenant(tenant_id):
            await CandidatesRepo(db).update_rescore(
                candidate_pk(stream.id, candidate),
                rescore_score=verdict.score,
                rescore_reason=verdict.reason,
                rescore_model=router_model_for(router, _PURPOSE, quality),
            )

        annotated = candidate.model_copy(
            update={
                "evidence": {
                    **candidate.evidence,
                    "rescore": verdict.model_dump(),
                }
            }
        )
        rescored.append(annotated)

    # Re-rank: rescored items first, sorted by rescore_score desc.
    rescored.sort(key=lambda c: float(c.evidence["rescore"]["score"]), reverse=True)
    remaining_top = top[len(rescored) :]
    final = rescored + remaining_top + rest

    if halted:
        with bound_tenant(tenant_id):
            await emit(db, "vision_rescore.budget_exhausted", {"stream_id": stream.id})

    return final, RescoreOutcome(
        rescored=len(rescored),
        skipped_budget=skipped_budget,
        skipped_cooldown=0,
        fatal_errors=fatal,
    )


async def _rescore_one(
    *,
    tenant_id: str,
    stream: Stream,
    candidate: Candidate,
    router: LLMRouter,
    governor: BudgetGovernor | None,
    frame_store: FrameStore,
    n_frames: int,
    quality: Quality | None,
) -> RescoreVerdict:
    """Sample N frames + call the vision LLM under the concurrency semaphore."""
    images = await asyncio.to_thread(
        _gather_frames,
        stream=stream,
        candidate=candidate,
        frame_store=frame_store,
        n_frames=n_frames,
    )
    user = _user_prompt_for(candidate, n_frames=n_frames)

    if governor is not None:
        async with governor.acquire_rescore_slot(tenant_id):
            return await router.complete_multimodal(
                tenant_id=tenant_id,
                purpose=_PURPOSE,
                system=_RESCORE_SYSTEM_PROMPT,
                user=user,
                images=images,
                schema=RescoreVerdict,
                quality=quality,
            )
    return await router.complete_multimodal(
        tenant_id=tenant_id,
        purpose=_PURPOSE,
        system=_RESCORE_SYSTEM_PROMPT,
        user=user,
        images=images,
        schema=RescoreVerdict,
        quality=quality,
    )


def _gather_frames(
    *,
    stream: Stream,
    candidate: Candidate,
    frame_store: FrameStore,
    n_frames: int,
) -> list[MultimodalImage]:
    """Sample `n_frames` around the candidate ts, caching via the frame_store.

    Cache key is `(stream_id, source_ts)` so two candidates sitting at the
    same anchor share frames, and a router-side retry doesn't re-decode.
    """
    if n_frames == 1:
        offsets = [0.0]
    else:
        step = _DEFAULT_FRAME_SPREAD_S / (n_frames - 1)
        offsets = [-(_DEFAULT_FRAME_SPREAD_S / 2.0) + i * step for i in range(n_frames)]

    images: list[MultimodalImage] = []
    missing_offsets: list[float] = []
    for offset in offsets:
        source_ts = max(0.0, candidate.timestamp + offset)
        cached = frame_store.get(stream.id, source_ts)
        if cached is not None:
            images.append(MultimodalImage(media_type="image/jpeg", data=cached))
        else:
            missing_offsets.append(offset)

    if missing_offsets:
        # One sample_frames call gets all missing offsets in one cap.read pass.
        sampled = sample_frames(
            stream.source_video_path,
            ts=candidate.timestamp,
            n=len(missing_offsets),
            spread_s=_DEFAULT_FRAME_SPREAD_S
            if len(missing_offsets) > 1
            else 0.0,
        )
        for offset, blob in zip(missing_offsets, sampled, strict=False):
            source_ts = max(0.0, candidate.timestamp + offset)
            frame_store.put(stream.id, source_ts, blob)
            images.append(MultimodalImage(media_type="image/jpeg", data=blob))

    return images


def router_model_for(router: LLMRouter, purpose: str, quality: Quality | None) -> str:
    """Resolve the model name the router would pick for `(purpose, quality)`.

    Surfaced as `candidates.rescore_model` so the dashboard can show which
    model issued each verdict (cheap haiku rescores != opus rescores).
    """
    rule = router._config.routing.get(purpose)
    if rule is None:
        return "unknown"
    effective: Quality = quality or rule.default_quality
    return router._config.model_for(rule.primary, effective)


# Silence the unused-import warning for `_dt` (kept as a convenient seam if
# the cooldown logic ever needs to reach `clock` from this module).
_ = _dt
