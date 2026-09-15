"""Viral-hook variant generator — voice-markers spec slice F.7-B.

This generator's only job is to produce candidate TITLE LINES — the
bold-on-white text that sits at the top of the rendered clip and drives
the swipe rate. (The old persona-voice caption generator was removed.)

Pipeline:
    persona context + clip transcript snippet + tone preset
        ↓
    LLMRouter.complete(purpose="hook_generation", schema=HookBatch)
        ↓
    list[Hook]  (5 short, punchy, swipe-stopping titles)

The tone preset rotates the prompt voice without burning a separate
LLM purpose for each. Five presets ship in F.7-B:

  * default      — generic creator-tone, the safe default
  * aggressive   — confrontational, no-punches-pulled
  * gen_z        — TikTok-native slang, abbreviations, emoji use
  * corporate    — clean, professional, no slang or emoji
  * curious      — open-loop questions that demand the answer

Stays under the existing budget governor — every call routes
through `LLMRouter.complete` with `purpose="hook_generation"`,
which the router config maps to a default quality + provider chain.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from chalybclip.llm import LLMRouter

ToneId = Literal["default", "aggressive", "gen_z", "corporate", "curious"]

# Default count — five gives the operator a real choice without
# blowing the budget. The router caps this; the schema enforces it.
DEFAULT_N = 5
MIN_N = 1
MAX_N = 10


class Hook(BaseModel):
    """One generated title line."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=140)


class HookBatch(BaseModel):
    """Structured response from the hook-generation purpose."""

    model_config = ConfigDict(extra="forbid")
    hooks: list[Hook] = Field(min_length=1, max_length=MAX_N)


# ---- prompt assembly ----


_TONE_PROMPTS: dict[ToneId, str] = {
    "default": (
        "Write in the streamer's natural voice — punchy, short, no fluff. "
        "Avoid clickbait that misrepresents the moment."
    ),
    "aggressive": (
        "Write CONFRONTATIONAL titles. Pick a fight with the viewer's "
        "expectations. Conflict, status, callout. No emoji. ALL-CAPS for "
        "the most charged word, sparingly."
    ),
    "gen_z": (
        "Write TikTok-native Gen Z titles. Lowercase, abbreviated, slang "
        "(fr, ngl, lowkey, no cap, ate, ratio'd). One emoji max. "
        "Concise — under 8 words."
    ),
    "corporate": (
        "Write CLEAN, PROFESSIONAL titles. No slang, no emoji, no "
        "ALL-CAPS. Could appear on a brand's official social channel "
        "without seeming off-brand."
    ),
    "curious": (
        "Write OPEN-LOOP titles that demand resolution — questions, "
        "cliffhangers, partial reveals. The viewer should NEED to watch "
        "to find out the answer. No spoilers."
    ),
}


_SYSTEM_PROMPT = """\
You write titles for a Spanish-speaking streamer's clip channel
(TikTok / Reels / Shorts). You are not a copywriter — you run clip
channels for a living. Your titles read like the ones on the channels
that clip Ibai, Spreen or ElXokas: native to the culture, never
corporate, never sounding like a translation from English.

You receive:
  - the streamer's persona / voice (tone reference ONLY)
  - context about the clip (stream title + why it was clipped)
  - a transcript snippet from the clip (may be a placeholder)
  - the target language
  - N, the number of title candidates to write

Return exactly N candidates as a JSON object matching the HookBatch
schema:
    { "hooks": [ { "text": "..." }, { "text": "..." }, ... ] }

THE REGISTER (shown in Spanish; if the target language is not Spanish,
reproduce the same clip-culture energy natively in that language —
never translate these literally):
  - Jerga real de directo: "se pasó", "no puede ser", "se vino abajo",
    "lo deja sin palabras", "explotó el chat". Cero tono de marca.
  - MAYÚSCULAS selectivas: only the single most charged word or short
    phrase goes in caps ("lo dijo EN SERIO"). Never the whole title.
  - Emojis con intención: 0-2 max, placed where they punch
    (😭 💀 🔥 ⚽ 😱 🎮). Decorative emoji spam reads as spam.
  - Community texture when the clip supports it: "el chat",
    "en directo", "en vivo", "clip del año".
  - Prefer present tense and immediacy over past-tense reporting.

THE OPEN LOOP: every title withholds exactly one thing — name the
reaction and hide the cause, OR name the cause and hide the outcome.
Never reveal both; never hide both.

HARD RULES:
  - Max 90 characters per title. The best titles are 30-60 characters —
    the text burns onto the video as a single line.
  - TRUTH LOCK: dramatize only what the transcript and context actually
    contain. NEVER invent quotes, scores, results, names, fights,
    accusations, or "exposed"-style claims the inputs do not support.
    Never accuse anyone, named or implied, of cheating, lying, crime,
    or misconduct; keep the target exactly as unnamed as the transcript
    keeps it. No fabricated consequences. If the context says two teams
    played, you do not know who scored — write the roar, not an
    invented goal. Fabricated drama about real people is a defamation
    risk and kills the channel.
  - Never put the streamer's name, persona name, or channel name in a
    title.
  - No hashtags. No quotation marks around the title. No trailing
    period.
  - Never repeat the transcript verbatim — reframe it into a hook.
  - No meta-commentary about your inputs: never mention a missing
    transcript, "no captions", guessing, or betting on what happened.
    Every title must read like you watched the clip three times.
  - When the transcript is a placeholder, hook off the CONTEXT: a
    stream titled "México 1 - 0 Corea" yields football-roar titles,
    never filler about missing information.

Make the N candidates genuinely different angles — each must pull a
DIFFERENT tension lever: incredulity, open loop, community reaction,
stakes, "you had to be there". The operator picks one.
"""


def _user_prompt(
    *,
    persona_voice: str,
    persona_language: str,
    transcript_snippet: str,
    tone: ToneId,
    n: int,
    clip_context: str = "",
    avoid_hooks: tuple[str, ...] = (),
) -> str:
    has_snippet = bool(transcript_snippet.strip())
    snippet_block = (
        transcript_snippet.strip()
        if has_snippet
        # No defeatist placeholder — the system prompt forbids writing
        # about a missing transcript, so steer the model to the context.
        else "(transcript unavailable — write the hooks from the context above)"
    )
    persona_block = persona_voice.strip() or "(no persona voice provided)"
    # Always present so the TRUTH LOCK has an explicit scope — when the
    # caller has no context, the placeholder anchors the model to the
    # transcript instead of letting it invent a premise.
    context_block = clip_context.strip() or "(only the transcript below is available)"
    # The register baked into the system prompt IS the default voice —
    # a tone preset is an operator override, appended only when chosen.
    tone_block = ""
    if tone != "default" and tone in _TONE_PROMPTS:
        tone_block = f"Extra tone instruction: {_TONE_PROMPTS[tone]}\n\n"
    # Cross-clip uniqueness: hooks already burned onto sibling clips of
    # this stream. The model must not converge on the same line twice.
    avoid_block = ""
    used = [h.strip() for h in avoid_hooks if h.strip()]
    if used:
        listing = "\n".join(f"  - {h}" for h in used)
        avoid_block = (
            "Titles already used on other clips from this same stream — "
            "every new candidate must be clearly different from ALL of "
            f"these:\n{listing}\n\n"
        )
    return (
        "Streamer persona / voice (tone reference only — never name them "
        f"in a title):\n{persona_block}\n\n"
        f"What we know about this clip:\n{context_block}\n\n"
        f"Language to write in: {persona_language}\n\n"
        f"{tone_block}"
        f"{avoid_block}"
        f"Transcript snippet from the clip:\n\"\"\"\n{snippet_block}\n\"\"\"\n\n"
        f"Generate {n} title candidates for this clip. "
        "Dramatize the real moment — never invent one."
    )


# ---- public entry ----


async def generate_hooks(
    *,
    tenant_id: str,
    persona_voice: str,
    persona_language: str,
    transcript_snippet: str,
    clip_context: str = "",
    tone: ToneId = "default",
    n: int = DEFAULT_N,
    avoid_hooks: tuple[str, ...] = (),
    router: LLMRouter,
) -> list[Hook]:
    """Generate `n` viral-hook title candidates for one clip.

    Args:
        tenant_id: Bound tenant — passed through to the router for
            cost-tracking attribution.
        persona_voice: The streamer's voice prompt (free-form text
            describing how they talk).
        persona_language: ISO 639-1 (`es`, `en`, ...) the hooks
            should be written in.
        transcript_snippet: 1-3 sentences from the clip's transcript,
            used as the hook's context. Empty string is allowed — the
            LLM then writes from `clip_context` instead of inventing
            meta-commentary about the missing transcript.
        clip_context: extra context the model should hook off of when
            the transcript is thin/empty — typically the stream title
            (e.g. "Mexico 1 - 0 Korea") plus the detection reason.
        tone: One of the five ToneId presets. Falls back to "default"
            if the caller passes an unrecognized id.
        n: Number to generate. Clamped to [MIN_N, MAX_N].
        avoid_hooks: Hooks already used on sibling clips of this
            stream — listed in the prompt as forbidden so a batch
            never ships the same line twice.
        router: Configured LLMRouter — purpose `hook_generation`
            must be in its routing rules.

    Returns:
        list of `Hook` objects, length `n` (or whatever the LLM
        actually returned, validated against the schema).
    """
    n_clamped = max(MIN_N, min(MAX_N, n))
    user = _user_prompt(
        persona_voice=persona_voice,
        persona_language=persona_language,
        transcript_snippet=transcript_snippet,
        clip_context=clip_context,
        tone=tone,
        n=n_clamped,
        avoid_hooks=avoid_hooks,
    )
    batch = await router.complete(
        tenant_id=tenant_id,
        purpose="hook_generation",
        system=_SYSTEM_PROMPT,
        user=user,
        schema=HookBatch,
    )
    return list(batch.hooks)


def tone_choices() -> list[tuple[ToneId, str]]:
    """(id, label) list for the editor's tone-picker dropdown."""
    return [
        ("default", "Default — streamer's natural voice"),
        ("aggressive", "Aggressive — confrontational, no-punches-pulled"),
        ("gen_z", "Gen Z — TikTok-native, lowercase, slang"),
        ("corporate", "Corporate — clean, professional, no slang"),
        ("curious", "Curious — open-loop questions, cliffhangers"),
    ]
