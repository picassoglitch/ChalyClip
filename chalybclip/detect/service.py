"""Trigger detection — voice-phrase + chat-heat fan-in.

Detectors:
1. Voice triggers (existing Phase 0 logic, internal to this file): slide
   each configured phrase across the Whisper transcript and emit a hit
   when Levenshtein <= `fuzzy_distance`.
2. Chat heat (Phase 1, in `chat_heat.py`): rolling-baseline spike test
   on chat-replay msg/sec.

`detect_candidates(...)` is the public entry point that runs both detectors
and merges their output. Adjacent candidates within `merge_window_s` get
clustered; the highest-scoring per cluster wins, and per-signal evidence
unions under `evidence["matches"]` so reviewers can see which detectors
fired together.

Phase 1 ships voice + chat. Audio energy lands in Task 4; visual signals
in Tasks 5-6. All three plug into the same fusion path.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from chalybclip.config import DetectionConfig
from chalybclip.errors import DetectionError
from chalybclip.ingest import ChatReplay, Stream
from chalybclip.transcribe import Transcript
from chalybclip.vision import VisualSignalTrack

from .audio_energy import detect_audio_energy
from .chat_heat import detect_chat_heat
from .fusion import (
    FusionBonuses,
    FusionConfig,
    FusionWeights,
    fuse_candidates,
)
from .levenshtein import levenshtein
from .models import Candidate, CandidateBatch
from .viral import detect_viral_moments
from .visual_signals import detect_visual_candidates


@dataclass(frozen=True)
class _FlatWord:
    text: str
    ts: float
    end_ts: float
    prob: float


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Accents are preserved — `clipéalo` and `clipealo` should match via the
    fuzzy distance, not via aggressive normalization that loses signal.
    """
    text = unicodedata.normalize("NFC", text)
    text = _PUNCT_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", text).strip()


def _flatten(transcript: Transcript) -> list[_FlatWord]:
    flat: list[_FlatWord] = []
    for seg in transcript.segments:
        for w in seg.words:
            flat.append(_FlatWord(text=w.text, ts=w.ts, end_ts=w.end_ts, prob=w.prob))
    return flat


def detect_voice_triggers(
    tenant_id: str,
    stream: Stream,
    transcript: Transcript,
    config: DetectionConfig,
    *,
    diarization: object | None = None,
    extra_phrases_by_speaker: dict[str, object] | None = None,
    extra_phrases_tenant_wide: object | None = None,
) -> list[Candidate]:
    """Return merged voice-trigger candidates for `stream`.

    All inputs must agree on `tenant_id` (CLAUDE.md hard rule #1).

    `diarization` is optional. When supplied (a `chalybclip.diarize.Diarization`
    with `skipped=False`), each emitted candidate carries a `speaker_label`
    in evidence, and the cooldown is applied per (speaker, trigger_kind)
    instead of globally — so a co-host firing 'clipea esto' 1s after the
    host still produces a separate clip. The argument is typed as `object`
    here to avoid importing chalybclip.diarize at module top (would create a
    cycle: pipeline → detect → diarize → db, all already touched).

    `extra_phrases_by_speaker` lets each speaker contribute their brand-kit's
    custom trigger phrases on top of the tenant base list. Map shape:
        {speaker_label: CustomTriggerPhrases(forward=[...], retroactive=[...])}
    Typed as `dict[str, object]` here for the same reason as `diarization` —
    the chalybclip.db.models import would form a back-edge through the db
    package. The values must duck-type to `.forward` + `.retroactive`
    (list[str] each); the resolver in `chalybclip.branding.service` produces
    them. Extra-phrase hits are restricted to candidates whose attributed
    speaker matches the map key — a co-host's kit can't accidentally
    create candidates for words the host said.

    `extra_phrases_tenant_wide` (slice O.39) is the solo-streamer
    counterpart: a single CustomTriggerPhrases (typed as object for the
    same reason as the map above) applied with NO speaker restriction.
    The pipeline supplies the tenant default brand kit's phrases here
    when diarization was skipped — there are no speaker labels to key
    off, so we just fire the kit on anyone. Quietly ignored when
    diarization succeeded; per-speaker mapping above handles that case.
    """
    if tenant_id != stream.tenant_id:
        raise DetectionError(f"tenant mismatch: caller={tenant_id!r}, stream={stream.tenant_id!r}")
    if tenant_id != transcript.tenant_id:
        raise DetectionError(
            f"tenant mismatch: caller={tenant_id!r}, transcript={transcript.tenant_id!r}"
        )
    if stream.id != transcript.stream_id:
        raise DetectionError(
            f"stream/transcript mismatch: stream={stream.id} transcript={transcript.stream_id}"
        )

    voice_cfg = config.voice
    if not voice_cfg.enabled:
        return []

    flat = _flatten(transcript)
    if not flat:
        return []

    raw: list[Candidate] = []
    # Forward triggers — clip extends forward from the timestamp.
    _scan_phrase_family(
        flat=flat,
        phrases_by_lang=voice_cfg.phrases,
        fuzzy_distance=voice_cfg.fuzzy_distance,
        weight=voice_cfg.weight,
        kind="forward",
        out=raw,
        diarization=diarization,
    )
    # Retroactive triggers — clip extends BACKWARD from the timestamp.
    _scan_phrase_family(
        flat=flat,
        phrases_by_lang=voice_cfg.retroactive_phrases,
        fuzzy_distance=voice_cfg.fuzzy_distance,
        weight=voice_cfg.weight,
        kind="retroactive",
        out=raw,
        retroactive_lookback_s=voice_cfg.retroactive_lookback_s,
        diarization=diarization,
    )
    # Per-kit additions — each speaker's brand kit contributes additional
    # forward/retroactive phrases. Each scan is restricted to candidates
    # whose attributed speaker matches the map key, so a co-host's kit
    # can't fire on words the host said. The detector cooldown step
    # below then dedupes anything that also matched a base phrase.
    if extra_phrases_by_speaker:
        for speaker_label, phrases in extra_phrases_by_speaker.items():
            kit_forward = list(getattr(phrases, "forward", []) or [])
            kit_retroactive = list(getattr(phrases, "retroactive", []) or [])
            if not kit_forward and not kit_retroactive:
                continue
            # Language is unknown for kit-added phrases; treat them as
            # "any language" by routing all of them through the 'es' bucket,
            # which is the only language the detector currently scans
            # word-for-word. Future i18n: kit phrases gain an explicit lang.
            if kit_forward:
                _scan_phrase_family(
                    flat=flat,
                    phrases_by_lang={"es": kit_forward},
                    fuzzy_distance=voice_cfg.fuzzy_distance,
                    weight=voice_cfg.weight,
                    kind="forward",
                    out=raw,
                    diarization=diarization,
                    restrict_to_speaker=speaker_label,
                )
            if kit_retroactive:
                _scan_phrase_family(
                    flat=flat,
                    phrases_by_lang={"es": kit_retroactive},
                    fuzzy_distance=voice_cfg.fuzzy_distance,
                    weight=voice_cfg.weight,
                    kind="retroactive",
                    out=raw,
                    retroactive_lookback_s=voice_cfg.retroactive_lookback_s,
                    diarization=diarization,
                    restrict_to_speaker=speaker_label,
                )

    # Slice O.39 — tenant-wide kit phrases (solo-streamer path). When
    # diarization is skipped, the per-speaker map above is empty by
    # construction (no speaker labels exist). The pipeline supplies the
    # tenant default brand kit's phrases here so the operator's setup
    # ("clipea esto" + their own preferred phrases) still fires on every
    # word in the VOD. No speaker restriction = anyone's words can hit.
    if extra_phrases_tenant_wide is not None:
        tw_forward = list(getattr(extra_phrases_tenant_wide, "forward", []) or [])
        tw_retroactive = list(
            getattr(extra_phrases_tenant_wide, "retroactive", []) or []
        )
        if tw_forward:
            _scan_phrase_family(
                flat=flat,
                phrases_by_lang={"es": tw_forward},
                fuzzy_distance=voice_cfg.fuzzy_distance,
                weight=voice_cfg.weight,
                kind="forward",
                out=raw,
                diarization=diarization,
            )
        if tw_retroactive:
            _scan_phrase_family(
                flat=flat,
                phrases_by_lang={"es": tw_retroactive},
                fuzzy_distance=voice_cfg.fuzzy_distance,
                weight=voice_cfg.weight,
                kind="retroactive",
                out=raw,
                retroactive_lookback_s=voice_cfg.retroactive_lookback_s,
                diarization=diarization,
            )

    # Per-speaker cooldown — keeps the first trigger of each
    # (speaker, kind) within `cooldown_s`. Falls back to a global
    # cooldown when no diarization was attached.
    pruned = _apply_per_speaker_cooldown(raw, cooldown_s=voice_cfg.cooldown_s)
    return _merge_candidates(pruned, window_s=config.merge_window_s)


def _scan_phrase_family(
    *,
    flat: list[_FlatWord],
    phrases_by_lang: dict[str, list[str]],
    fuzzy_distance: int,
    weight: float,
    kind: str,
    out: list[Candidate],
    retroactive_lookback_s: float | None = None,
    diarization: object | None = None,
    restrict_to_speaker: str | None = None,
) -> None:
    """Append candidates for one phrase family (forward OR retroactive) to `out`.

    The two families share the entire matching algorithm; only the
    `evidence['trigger_kind']` flag and the optional retroactive metadata
    differ. Cut step reads `trigger_kind` to decide the window direction.

    When `diarization` is provided and `.overlap_speaker(...)` returns a
    label, the candidate carries `evidence['speaker_label']` so the
    per-speaker cooldown + downstream brand-kit-by-speaker resolution
    can use it.

    `restrict_to_speaker` is the per-kit-phrases filter (slice C.2). When
    set, ONLY emissions whose attributed speaker_label matches the
    argument survive — a co-host's kit can't add candidates on the host's
    words.
    """
    for language, phrases in phrases_by_lang.items():
        for phrase in phrases:
            phrase_norm = _normalize(phrase)
            if not phrase_norm:
                continue
            phrase_tokens = phrase_norm.split()
            window_len = len(phrase_tokens)
            if window_len == 0 or window_len > len(flat):
                continue
            for i in range(len(flat) - window_len + 1):
                window = flat[i : i + window_len]
                joined = " ".join(_normalize(w.text) for w in window).strip()
                if not joined:
                    continue
                dist = levenshtein(phrase_norm, joined, max_dist=fuzzy_distance)
                if dist > fuzzy_distance:
                    continue
                confidence = sum(w.prob for w in window) / len(window)
                score = weight * confidence
                snippet = " ".join(w.text.strip() for w in window)
                evidence: dict[str, object] = {
                    "phrase": phrase,
                    "language": language,
                    "transcript_snippet": snippet,
                    "distance": dist,
                    "confidence": confidence,
                    "end_ts": window[-1].end_ts,
                    "trigger_kind": kind,
                }
                if kind == "retroactive" and retroactive_lookback_s is not None:
                    evidence["retroactive_lookback_s"] = retroactive_lookback_s
                # Attribute to a speaker if diarization is available and the
                # turn covers (or mostly overlaps) the phrase span.
                speaker_label = _attribute_speaker(
                    diarization, window[0].ts, window[-1].end_ts
                )
                if restrict_to_speaker is not None and speaker_label != restrict_to_speaker:
                    # Kit-added phrase fired on the wrong speaker — drop it.
                    continue
                if speaker_label is not None:
                    evidence["speaker_label"] = speaker_label
                out.append(
                    Candidate(
                        timestamp=window[0].ts,
                        score=score,
                        reason="voice",
                        evidence=evidence,
                    )
                )


def _attribute_speaker(
    diarization: object | None, ts: float, end_ts: float
) -> str | None:
    """Best-effort speaker attribution; returns None when diarization is absent
    or doesn't cover this phrase span.

    Typed loosely as `object` to keep the diarize package decoupled from
    detect at module-load time. Falls back to None if the dynamic call
    fails for any reason — never raises into the detector."""
    if diarization is None:
        return None
    overlap = getattr(diarization, "overlap_speaker", None)
    skipped = getattr(diarization, "skipped", True)
    if not callable(overlap) or skipped:
        return None
    try:
        label = overlap(ts, end_ts)
    except Exception:  # noqa: BLE001
        return None
    return label if isinstance(label, str) else None


def _apply_per_speaker_cooldown(
    candidates: list[Candidate], *, cooldown_s: float
) -> list[Candidate]:
    """Drop later candidates within `cooldown_s` of an earlier same-kind hit.

    Cooldown key is (speaker_label, trigger_kind). When speaker_label is
    absent (no diarization, or speaker unattributable on a given hit), all
    no-speaker candidates of the same kind share one key — equivalent to
    the global cooldown the spec describes when diarization is off.

    Stable order: candidates are walked in ascending-timestamp order, the
    first wins per cooldown bucket. Always preserves at least the first
    hit per (speaker, kind).
    """
    if cooldown_s <= 0.0 or not candidates:
        return list(candidates)
    sorted_c = sorted(candidates, key=lambda c: c.timestamp)
    kept: list[Candidate] = []
    last_ts_by_key: dict[tuple[str | None, str], float] = {}
    for c in sorted_c:
        ev = c.evidence or {}
        speaker = ev.get("speaker_label")
        speaker_key: str | None = speaker if isinstance(speaker, str) else None
        kind_raw = ev.get("trigger_kind", "forward")
        kind = str(kind_raw) if kind_raw is not None else "forward"
        key = (speaker_key, kind)
        last_ts = last_ts_by_key.get(key)
        if last_ts is not None and c.timestamp - last_ts < cooldown_s:
            continue
        kept.append(c)
        last_ts_by_key[key] = c.timestamp
    return kept


def _merge_candidates(candidates: list[Candidate], *, window_s: float) -> list[Candidate]:
    """Collapse temporally-close candidates via the slice-G.1 weighted fusion.

    Kept under the historical name so existing test suites that import
    `_merge_candidates` directly keep working. The implementation now
    routes through `fuse_candidates` with default weights — operators
    can override via `DetectionConfig.fusion` and the cluster window
    via `window_s` (mapped to `cluster_window_s`).
    """
    return fuse_candidates(
        candidates,
        config=FusionConfig(cluster_window_s=window_s),
    )


def detect_candidates(
    tenant_id: str,
    stream: Stream,
    transcript: Transcript,
    config: DetectionConfig,
    *,
    chat_replay: ChatReplay | None = None,
    visual_track: VisualSignalTrack | None = None,
    viral_candidates: list[Candidate] | None = None,
    diarization: object | None = None,
    extra_phrases_by_speaker: dict[str, object] | None = None,
    extra_phrases_tenant_wide: object | None = None,
) -> list[Candidate]:
    """Run every available detector and return the fused candidate stream.

    Detectors:
        * voice triggers — fuzzy phrase match on the transcript
        * chat heat — rolling-baseline spike on msg/sec (when chat_replay set)
        * audio energy — RMS spike on the source audio (when enabled)
        * visual signals — scene cuts / faces / motion (when visual_track set)
        * viral — pass-through; the caller runs `detect_viral_moments()`
          (async, needs an LLMRouter) and forwards the result here. This
          keeps the sync vs async boundary clean — `detect_candidates` stays
          sync and pure; the pipeline awaits the LLM call once and threads
          the candidates in.

    Each detector emits its own Candidates; this function concatenates them
    and runs `_merge_candidates` so hits from different signals within the
    same window collapse into one cluster with `evidence["matches"]`
    listing each source.
    """
    voice = detect_voice_triggers(
        tenant_id=tenant_id,
        stream=stream,
        transcript=transcript,
        config=config,
        diarization=diarization,
        extra_phrases_by_speaker=extra_phrases_by_speaker,
        extra_phrases_tenant_wide=extra_phrases_tenant_wide,
    )
    chat: list[Candidate] = []
    if chat_replay is not None:
        chat = detect_chat_heat(
            tenant_id=tenant_id,
            stream=stream,
            chat_replay=chat_replay,
            config=config.chat_heat,
        )
    audio: list[Candidate] = detect_audio_energy(
        tenant_id=tenant_id, stream=stream, config=config.audio_energy
    )
    visual: list[Candidate] = []
    if visual_track is not None:
        visual = detect_visual_candidates(
            tenant_id=tenant_id,
            stream=stream,
            track=visual_track,
            config=config.visual,
        )
    viral = viral_candidates or []
    if not chat and not audio and not visual and not viral:
        # Voice-only — already merged by detect_voice_triggers.
        return voice
    # Slice G.1 — weighted fusion. Operators tune weights / bonuses via
    # `detection.fusion` in chalybclip.yaml; the dataclass below is the
    # runtime shape `fuse_candidates` consumes.
    fusion_cfg = FusionConfig(
        cluster_window_s=config.merge_window_s,
        overlap_window_s=config.fusion.overlap_window_s,
        weights=FusionWeights(
            voice=config.fusion.voice,
            visual=config.fusion.visual,
            audio=config.fusion.audio,
            chat=config.fusion.chat,
            viral=config.fusion.viral,
            transcript_hook=config.fusion.transcript_hook,
        ),
        bonuses=FusionBonuses(
            two_detectors=config.fusion.two_detector_bonus,
            three_plus_detectors=config.fusion.three_plus_detector_bonus,
            face_visible=config.fusion.face_visible_bonus,
            strong_signal=config.fusion.strong_signal_bonus,
        ),
    )
    return fuse_candidates(
        voice + chat + audio + visual + viral,
        config=fusion_cfg,
    )


def fallback_interval_candidates(
    tenant_id: str,
    stream: Stream,
    *,
    max_clips: int = 5,
    target_clip_s: float = 30.0,
) -> list[Candidate]:
    """Last-resort candidates when NO detector fired.

    A silent, static clip (no speech, no audio peaks, no visual motion)
    produces zero candidates from every signal — but the product promise is
    "give me clips". So we place evenly-spaced anchors across the video and
    let `cut` window around them, guaranteeing the user always has clips to
    review. Score is intentionally low (0.1) so any real signal-based
    candidate outranks these.

    One anchor per ~`target_clip_s` of video, capped at `max_clips`, each
    centered in its slice. Returns [] only for a zero-duration stream.
    """
    duration = float(getattr(stream, "duration_s", 0.0) or 0.0)
    if duration <= 0.0:
        return []
    n = max(1, min(max_clips, int(duration // target_clip_s) or 1))
    return [
        Candidate(
            timestamp=round(duration * (i + 0.5) / n, 3),
            score=0.1,
            reason="interval",
            evidence={"fallback": "interval", "note": "no detector fired"},
        )
        for i in range(n)
    ]


def save_candidates(stream_dir: Path, batch: CandidateBatch) -> Path:
    """Persist candidates to `<stream_dir>/candidates.json`."""
    out = Path(stream_dir) / "candidates.json"
    out.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
    return out


def load_candidates(stream_dir: Path) -> CandidateBatch:
    """Read candidates back from disk."""
    path = Path(stream_dir) / "candidates.json"
    if not path.exists():
        raise DetectionError(f"candidates not found at {path}")
    return CandidateBatch.model_validate_json(path.read_text("utf-8"))
