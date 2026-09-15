"""YAML config loader.

Per CLAUDE.md the layered config order is: defaults → YAML → environment.
The YAML side lives here; environment variables live in `chalybclip.settings`.

Phase 0 only models the sections that have functional code consuming them
(detection). Everything else maps with `extra="allow"` so the same file
serves later phases without a schema version bump.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from chalybclip.errors import ChalybClipError


class VoiceDetectorConfig(BaseModel):
    """Voice trigger phrase list + fuzzy match params.

    Two phrase families with different clip-window semantics:

    * `phrases` — forward triggers. The streamer says the phrase BEFORE the
      moment ('watch this — clipea esto'). The cut step extends forward
      from the trigger timestamp by `clip.pre_roll_s + post_roll_s`.

    * `retroactive_phrases` — retroactive triggers. The streamer says the
      phrase AFTER the moment ('that was insane — clipeaste eso'). The
      cut step uses `retroactive_lookback_s` of audio BEFORE the trigger
      timestamp and ignores pre/post roll. This is the more natural case
      in live streaming.
    """

    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0)
    fuzzy_distance: int = Field(default=2, ge=0)
    # Slice O.41 — bake in the spec defaults instead of `{}`. Railway
    # boots without a chalybclip.yaml file (the YAML is operator-local
    # config), and the empty default meant the voice detector returned
    # zero candidates on every Spanish stream until the operator
    # discovered the YAML knob. Same list as the example YAML so
    # behavior is identical with-or-without a config file.
    phrases: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "es": [
                "clipea esto",
                "clipéalo",
                "saca un clip",
                "guarda esto",
                "momento clip",
                "este momento",
            ],
            "en": [
                "clip this",
                "clip that",
                "someone clip this",
                "did you clip that",
            ],
        },
        description="ISO 639-1 → list of forward trigger phrases.",
    )
    retroactive_phrases: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "es": [
                "clipeaste eso",
                "clipea eso",
                "clipearon eso",
                "eso fue épico",
            ],
            "en": [
                "did you clip that",
                "clip that",
                "tell me you clipped that",
                "please tell me you got that",
            ],
        },
        description="ISO 639-1 → list of retroactive trigger phrases.",
    )
    retroactive_lookback_s: float = Field(
        default=60.0,
        gt=0.0,
        description="When a retroactive phrase fires, the clip covers this "
        "many seconds BEFORE the timestamp.",
    )
    cooldown_s: float = Field(
        default=2.5,
        ge=0.0,
        description="Minimum gap between two triggers of the same kind. "
        "Only there to drop stutter-dupes inside a single utterance "
        "('clip— clipea esto'). Operator-flagged the old 10s default: "
        "if they say the phrase three times in two minutes they want "
        "three clips, not one.",
    )


class ChatHeatConfig(BaseModel):
    """Chat heat detector — spike when msg/sec >> rolling baseline."""

    enabled: bool = False
    weight: float = Field(default=0.7, ge=0.0)
    baseline_window_s: float = Field(default=300.0, gt=0.0)
    spike_ratio: float = Field(default=3.0, gt=0.0)
    absolute_floor_msg_per_s: float = Field(default=5.0, ge=0.0)


class AudioEnergyConfig(BaseModel):
    """Audio energy detector — spike when frame RMS >> rolling baseline,
    sustained for at least `sustain_s` to suppress one-frame pops.
    """

    enabled: bool = False
    weight: float = Field(default=0.5, ge=0.0)
    frame_s: float = Field(default=0.5, gt=0.0, description="RMS bin size in seconds.")
    baseline_window_s: float = Field(default=300.0, gt=0.0)
    spike_ratio: float = Field(default=2.5, gt=0.0)
    sustain_s: float = Field(default=1.5, ge=0.0)


class VisualConfig(BaseModel):
    """Visual detector — fuses scene cuts, emotion transitions, and motion
    spikes (from `analyze-video`'s VisualSignalTrack) into Candidates.
    """

    # Slice F.7-H — flipped to True by default. The detector adds
    # scene-cut + motion + face-emotion signals to the candidate set,
    # which is essential for clips that are visually dramatic but
    # don't have a loud voice / chat moment (silent dunks, jumpscares,
    # reaction shots). It's a one-time-per-stream CPU cost capped by
    # `timeout_s`, and the operator can set this False in chalybclip.yaml
    # if they're running on an underpowered host.
    enabled: bool = True
    weight: float = Field(default=0.6, ge=0.0, description="Outer multiplier on the visual score.")
    cut_weight: float = Field(default=1.0, ge=0.0)
    emotion_weight: float = Field(default=0.7, ge=0.0)
    motion_weight: float = Field(default=0.5, ge=0.0)
    motion_baseline_window_s: float = Field(default=30.0, gt=0.0)
    motion_spike_ratio: float = Field(default=2.0, gt=0.0)
    emotion_labels: list[str] = Field(
        default_factory=lambda: ["smile", "laugh", "shock"],
        description="Which face_emotion values count as 'strong' (worth firing on).",
    )
    timeout_s: float = Field(
        default=600.0,
        gt=0.0,
        description=(
            "Hard cap on analyze_video runtime in seconds (floor). The pipeline "
            "actually uses max(timeout_s, duration_s * analyze_video_timeout_multiplier) "
            "so this is just the minimum even for very short videos. Slice O.27 "
            "bumped the floor 120 → 600 because PySceneDetect on prod CPU runs "
            "near 1× realtime and the 120 floor was getting hit on 60-90s clips."
        ),
    )


class DiarizationConfig(BaseModel):
    """Speaker diarization (pyannote-3.1) settings.

    Slice O.29 — flipped to True by default. The pipeline already
    short-circuits gracefully when (a) pyannote.audio isn't installed
    or (b) HF_TOKEN isn't set — the step logs `diarize.skipped` with
    the reason and downstream code reads `diarization.skipped` to fall
    back to un-attributed candidates. So flipping the default is safe:
    setups that DO have pyannote+HF_TOKEN get the feature for free;
    setups that don't see the same skip-with-reason they saw before.

    `match_threshold` is the cosine-sim cutoff for matching a new VOD's
    speaker embedding against the tenant's persistent `speakers` table.
    Lower values are more permissive (more likely to merge two recordings
    of the same person into one identity); higher values demand cleaner
    matches.
    """

    enabled: bool = True
    model: str = Field(default="pyannote/speaker-diarization-3.1")
    device: str = Field(default="cuda")
    worker_timeout_s: float = Field(
        default=7200.0,
        gt=0.0,
        description="Hard kill for the pyannote subprocess. A CUDA stall "
        "hangs without exiting; past this the run degrades to "
        "skipped-with-reason instead of wedging the whole pipeline. "
        "Generous by default — multi-hour VODs on CPU are slow.",
    )
    match_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    min_speech_for_id_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Speakers with less than this much total speech in a VOD "
        "are not auto-matched against the persistent speakers table — too "
        "little signal to risk a wrong merge. Still recorded as VOD-scoped "
        "labels so the user can label them manually.",
    )
    # Task A2 — when the transcribe provider emits per-utterance
    # speaker labels natively (AssemblyAI with speaker_labels=true),
    # we don't need the GPU-bound pyannote pass: we derive the same
    # Diarization shape from the transcript's segments after the
    # transcribe step finishes. Set to "transcribe" to opt in. Default
    # stays "pyannote" so existing setups don't change behavior.
    #
    # The "transcribe" mode skips both the pyannote inference AND the
    # cross-video resolve_speakers step (no embeddings). The speakers
    # table is still populated with per-VOD labels via the persistence
    # path that runs after detect. Cross-video persistent identity is
    # deferred (TODO: re-add via Modal + embeddings later if it drives
    # conversion).
    source: str = Field(
        default="pyannote",
        description="`pyannote` or `transcribe`. See class docstring.",
    )


class ViralConfig(BaseModel):
    """Viral-moment detector — feeds the transcript to an LLM and asks it to
    identify the 5-15 most clip-worthy moments based on controversy, emotion,
    quotability, and shock value.

    Slice O.27 — enabled by default. The voice-trigger-only detector
    returns 0 candidates the moment a streamer doesn't yell "clip this"
    (i.e. essentially every real-world VOD). Without the viral detector
    on, the pipeline finishes with 0 candidates → 0 clips → the user
    sees "Pipeline complete" and nothing to do, with no hint why.
    Costs one LLM call per stream — acceptable for paid-tier users who
    pay for clips and not for LLM-call budgets.
    """

    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0)
    quality: str = Field(
        default="standard",
        description="LLM quality tier (standard | premium).",
    )
    max_moments: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Hard cap on how many moments the LLM can return.",
    )
    min_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Drop moments below this LLM score before fusing.",
    )


class FusionConfigModel(BaseModel):
    """Weighted multi-signal fusion — slice G.1.

    The dataclass-based `FusionConfig` in `chalybclip.detect.fusion` is
    the runtime contract; this Pydantic mirror exists so operators can
    tune weights / bonuses from `chalybclip.yaml` without touching code.
    The two are kept in sync via `to_runtime()` below.
    """

    voice: float = Field(default=0.35, ge=0.0, le=1.0)
    visual: float = Field(default=0.20, ge=0.0, le=1.0)
    audio: float = Field(default=0.15, ge=0.0, le=1.0)
    chat: float = Field(default=0.15, ge=0.0, le=1.0)
    viral: float = Field(default=0.10, ge=0.0, le=1.0)
    transcript_hook: float = Field(default=0.05, ge=0.0, le=1.0)

    two_detector_bonus: float = Field(default=0.05, ge=0.0, le=1.0)
    three_plus_detector_bonus: float = Field(default=0.10, ge=0.0, le=1.0)
    face_visible_bonus: float = Field(default=0.05, ge=0.0, le=1.0)
    strong_signal_bonus: float = Field(default=0.05, ge=0.0, le=1.0)

    overlap_window_s: float = Field(default=10.0, gt=0.0)


class DetectionConfig(BaseModel):
    """All detector configuration."""

    voice: VoiceDetectorConfig = Field(default_factory=VoiceDetectorConfig)
    chat_heat: ChatHeatConfig = Field(default_factory=ChatHeatConfig)
    audio_energy: AudioEnergyConfig = Field(default_factory=AudioEnergyConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
    viral: ViralConfig = Field(default_factory=ViralConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    fusion: FusionConfigModel = Field(default_factory=FusionConfigModel)
    merge_window_s: float = Field(default=30.0, ge=0.0)


class ClipConfig(BaseModel):
    """Cut + reformat parameters for the clip step."""

    pre_roll_s: float = Field(default=30.0, ge=0.0)
    post_roll_s: float = Field(default=15.0, ge=0.0)
    # Ceiling for substantive clip windows (story / default / retroactive).
    # Default 60.0 keeps current behavior EXACTLY (every band already caps at
    # 60s). Raise it (e.g. 75-90) to let substantive clips run past 60s so
    # they clear platforms that only pay ABOVE 60s — notably TikTok Creator
    # Rewards (videos must be >1 min to earn). Punchy bands (reaction/quote)
    # are never lifted: stretching a snappy moment tanks completion + reach.
    # The publish-time per-platform duration guard then auto-routes a longer
    # clip away from shorter surfaces (Bluesky 60s, IG/FB 90s).
    max_clip_duration_s: float = Field(default=60.0, ge=10.0)
    output_aspect: str = Field(default="9:16", description="Phase 0 only supports 9:16.")
    output_width: int = Field(default=1080, gt=0)
    output_height: int = Field(default=1920, gt=0)
    encoder: str = "libx264"
    # Slice O.49 — preset "veryfast" -> "fast", CRF 23 -> 19. Operator
    # report: downloaded clips look low quality. The reason: the cut
    # step is the FIRST of three lossy passes in the export chain
    # (cut -> Playwright WebM recording -> ffmpeg H.264 mux). At
    # veryfast / CRF 23 the cut alone produces visible blocking on
    # text + faces; "fast" + CRF 19 gives mathematically higher
    # quality at roughly the same wall time on Railway CPU
    # (~+15-20% encode time, ~+30% bitrate). The mux pass also got
    # CRF 17 + "medium" so the final MP4 doesn't double-lose
    # detail. See preview_recorder.py:_mux_video_and_audio for the
    # corresponding mux-side change.
    preset: str = "fast"
    crf: int = Field(default=19, ge=0, le=51)
    burn_captions: bool = False
    # Slice G.1 — dynamic per-candidate clip windowing. When True
    # (the default) and a transcript is available, each clip's
    # start/end snaps to sentence boundaries inside a kind-aware
    # band (reaction 10-22s / quote 12-25s / story 35-60s / etc).
    # Set False to fall back to the legacy pre_roll/post_roll
    # geometry on every clip (useful when transcripts are unreliable
    # or for debugging the static cut path).
    dynamic_windowing: bool = True
    # Task 1a — per-candidate cut work runs concurrently up to this
    # many threads at once. Default 3 balances event-loop
    # responsiveness against CPU/disk contention on a typical
    # 8–12 core box; lower it (1) to restore the pre-Task-1 serial
    # behavior if ffmpeg starts thrashing, raise it on a big NVENC
    # host where the GPU is the bottleneck and the CPU is idle.
    cut_concurrency: int = Field(default=3, ge=1, le=16)
    # Cap the per-encode CPU thread count for software (libx264) encodes.
    # ffmpeg/x264 default to `threads=auto` (~1.5x core count); with the
    # cut fan-out, an operator render, and the poll pipeline all encoding
    # at once, that thread multiplier exhausts the container's PID/thread
    # ceiling and `x264_encoder_open` fails with EAGAIN. Pinning a small
    # count keeps each encode light. 0 = let ffmpeg decide (legacy auto).
    # Ignored by the NVENC path (GPU-bound, not thread-bound).
    ffmpeg_threads: int = Field(default=2, ge=0, le=64)
    # Task 1b — at startup we probe `ffmpeg -encoders` for
    # h264_nvenc. When present AND `prefer_nvenc=True` AND the
    # operator hasn't overridden `encoder` to something other than
    # libx264, BOTH the accurate-seek cut and the 9:16 reformat
    # use NVENC (drops ~20–30s/clip encode → 2–3s on an RTX host).
    # Set False to force the libx264 path bit-for-bit — rollback
    # escape hatch if NVENC quality looks wrong on a given source.
    prefer_nvenc: bool = True
    # NVENC constant-quality target. Defaulted to match `crf` so
    # the NVENC path is visually equivalent to the libx264 fallback
    # at the same operator-perceived quality. The platforms re-encode
    # aggressively anyway. Only consulted when NVENC is picked.
    nvenc_cq: int = Field(default=19, ge=0, le=51)


class ChalybClipConfig(BaseModel):
    """Root config object loaded from `config/chalybclip.yaml`."""

    model_config = ConfigDict(extra="allow")

    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    clip: ClipConfig = Field(default_factory=ClipConfig)


_DEFAULT_CONFIG_PATH = Path("config/chalybclip.yaml")
_DEFAULT_EXAMPLE_PATH = Path("config/chalybclip.example.yaml")


def load_config(path: Path | None = None) -> ChalybClipConfig:
    """Load `config/chalybclip.yaml`, falling back to the example file.

    If neither file exists, return defaults — a fresh checkout still
    has a working detection config.
    """
    candidates: list[Path] = (
        [Path(path)] if path is not None else [_DEFAULT_CONFIG_PATH, _DEFAULT_EXAMPLE_PATH]
    )

    for candidate in candidates:
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ChalybClipError(f"failed to parse {candidate}: {e}") from e
            return ChalybClipConfig.model_validate(data)

    if path is not None:
        raise ChalybClipError(f"config file not found: {path}")
    return ChalybClipConfig()


@lru_cache(maxsize=1)
def get_config() -> ChalybClipConfig:
    """Cached default loader; tests can call `get_config.cache_clear()`."""
    return load_config()
