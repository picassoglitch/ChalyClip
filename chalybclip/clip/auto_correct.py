"""Auto-correct a freshly cut clip — slice G.6.

Apply the AI's findings at CREATION time (while the source VOD still
exists) so a clip arrives publish-ready by itself, and the dashboard only
has to *report* what was corrected instead of offering a "fix it" button.

Run, in order, against one clip:

  1. auto-trim around integrity issues (freeze / silence) — re-cut to the
     longest clean window. The trim stashes a pristine pre-trim copy so
     revert keeps working even after the VOD is pruned.
  2. apply non-destructive overlay fixes (captions / banner / safe zones).
  3. recompute publishability and promote `cut` -> `ready_for_review` when
     the corrected clip clears the publish-ready bar.

The human-readable summary is written to `<clip_dir>/auto_corrections.json`
— a filesystem sidecar that survives overlay re-saves and needs no schema
change — and emitted as a `clip.auto_corrected` event. The dashboard reads
the sidecar to render the "Corregido automáticamente" panel.

This module is the single business-logic entry point; the pipeline calls
it per clip after the cut step. It re-uses the same pure helpers the
dashboard's manual endpoints call (clip_breakdown / apply_ai_fixes /
compute_publishability / auto_trim_around_integrity), so the automatic and
manual paths can never drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from chalybclip.db import ClipsRepo, Database, EventsRepo
from chalybclip.errors import ClipError

from .ai_fixes import apply_ai_fixes
from .breakdown import clip_breakdown
from .publishability import compute_publishability
from .trim import auto_trim_around_integrity

_log = structlog.get_logger(__name__)

AUTO_CORRECTIONS_FILENAME = "auto_corrections.json"

# Spanish labels for the integrity-issue kinds the trim removes.
_ISSUE_KIND_ES: dict[str, str] = {
    "freeze": "imagen congelada",
    "silence": "silencio",
}


async def auto_correct_clip(
    db: Database,
    *,
    clip_id: str,
    source_platform: str | None = None,
    brand_kit_url: str | None = None,
) -> dict[str, object]:
    """Auto-apply trim + overlay fixes to `clip_id`; return a report dict.

    Best-effort by design: a failure in any single step is logged and
    skipped, never raised, so one bad clip can't break the pipeline batch.
    The returned dict is the same shape written to the sidecar / event.
    """
    clips_repo = ClipsRepo(db)
    clip = await clips_repo.get(clip_id)
    if clip is None:
        raise ClipError(f"clip {clip_id!r} not found")

    safe_zone_target = _safe_zone_of(clip)

    # --- BEFORE snapshot (for the score delta the UI shows) ----------
    breakdown = await clip_breakdown(db, clip_id)
    before_verdict = compute_publishability(
        breakdown=breakdown,
        overlay_config=_overlay_of(clip),
        safe_zone_platform=safe_zone_target,
    )

    corrections: list[dict[str, str]] = []

    # --- 1) auto-trim around integrity issues ------------------------
    trimmed = False
    if breakdown.integrity_issues:
        pre_trim_duration = breakdown.duration_s
        issues = list(breakdown.integrity_issues)
        try:
            trim_result = await auto_trim_around_integrity(
                db, clip_id=clip_id, integrity_issues=issues
            )
        except ClipError as e:
            _log.warning(
                "clip.auto_correct.trim_failed", clip_id=clip_id, reason=str(e)
            )
            trim_result = {"outcome": "error"}

        if trim_result.get("outcome") == "trimmed":
            trimmed = True
            corrections.append(
                {
                    "kind": "trim",
                    "label": _trim_label_es(
                        issues, trim_result, pre_trim_duration
                    ),
                }
            )
            # Bounds changed: reload the clip + recompute the breakdown so
            # the AFTER verdict scores the trimmed (clean) window.
            clip = await clips_repo.get(clip_id) or clip
            breakdown = await clip_breakdown(db, clip_id)

    # --- 2) non-destructive overlay fixes ----------------------------
    overlay_now = _overlay_of(clip)
    fixes = apply_ai_fixes(
        overlay_config=overlay_now,
        safe_zone_platform=safe_zone_target,
        brand_kit_url=brand_kit_url,
        source_platform=source_platform,
    )
    if fixes.fixes:
        await clips_repo.set_overlay_config(
            clip_id, overlay_config=fixes.new_overlay_config
        )
        overlay_now = fixes.new_overlay_config
        corrections.extend(
            {"kind": "overlay", "label": f.why} for f in fixes.fixes
        )

    # --- AFTER verdict + cache --------------------------------------
    after_verdict = compute_publishability(
        breakdown=breakdown,
        overlay_config=overlay_now,
        safe_zone_platform=safe_zone_target,
    )
    try:
        await clips_repo.set_publishability(
            clip_id,
            score=int(after_verdict.score),
            status=after_verdict.status,
        )
    except Exception:  # noqa: BLE001 — cache write must not block correction
        pass

    # --- promote cut -> ready_for_review when publish-ready ----------
    promoted = False
    if after_verdict.status == "publish_ready" and clip.status == "cut":
        try:
            await clips_repo.update_status(clip_id, status="ready_for_review")
            promoted = True
        except Exception as e:  # noqa: BLE001 — promotion is best-effort
            _log.warning(
                "clip.auto_correct.promote_failed", clip_id=clip_id, reason=str(e)
            )

    report: dict[str, object] = {
        "clip_id": clip_id,
        "corrections": corrections,
        "trimmed": trimmed,
        "score_before": int(before_verdict.score),
        "score_after": int(after_verdict.score),
        "status": after_verdict.status,
        "ready": after_verdict.status == "publish_ready",
        "promoted": promoted,
    }

    _write_sidecar(Path(clip.path).parent, report)
    try:
        await EventsRepo(db).emit(type="clip.auto_corrected", payload=report)
    except Exception:  # noqa: BLE001 — event log is observability, not a gate
        pass

    _log.info(
        "clip.auto_corrected",
        clip_id=clip_id,
        corrections=len(corrections),
        trimmed=trimmed,
        score_before=report["score_before"],
        score_after=report["score_after"],
        promoted=promoted,
    )
    return report


# ---- sidecar I/O (read by the dashboard clip-detail handler) ----


def load_auto_corrections(clip_dir: Path) -> dict[str, object] | None:
    """Read the auto-corrections sidecar next to a clip MP4, or None."""
    path = Path(clip_dir) / AUTO_CORRECTIONS_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a corrupt sidecar just hides the panel
        return None


def _write_sidecar(clip_dir: Path, report: dict[str, object]) -> None:
    try:
        (clip_dir / AUTO_CORRECTIONS_FILENAME).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 — sidecar is best-effort
        pass


# ---- helpers ----


def _overlay_of(clip: object) -> dict[str, object]:
    cfg = getattr(clip, "overlay_config", None)
    return cfg if isinstance(cfg, dict) else {}


def _safe_zone_of(clip: object) -> str:
    cfg = _overlay_of(clip)
    szp = cfg.get("safe_zone_platform")
    return szp if isinstance(szp, str) and szp else "tiktok"


def _trim_label_es(
    issues: list[dict[str, object]],
    trim_result: dict[str, object],
    pre_trim_duration: float,
) -> str:
    """Build the Spanish "we trimmed X" line shown to the operator."""
    kinds = {str(i.get("kind", "")) for i in issues}
    nice = sorted(_ISSUE_KIND_ES.get(k, k) for k in kinds if k)
    what = " y ".join(nice) if nice else "tramos con problemas"
    new_dur = float(trim_result.get("new_duration_s") or 0.0)
    removed = max(0.0, pre_trim_duration - new_dur)
    return (
        f"Quitamos {int(round(removed))}s de {what} y dejamos el mejor "
        f"tramo limpio ({int(round(new_dur))}s)."
    )


__all__ = [
    "AUTO_CORRECTIONS_FILENAME",
    "auto_correct_clip",
    "load_auto_corrections",
]
