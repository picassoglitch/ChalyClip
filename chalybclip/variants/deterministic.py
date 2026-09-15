"""Deterministic (no-LLM) hook generation — titles from what we already have.

Hook generation is an LLM call, and LLM calls fail for boring, durable
reasons: the billing cap is hit, the key is revoked, the provider is down.
When that happened the pipeline shipped clips with NO hook, which cascaded
into degenerate one-word captions and dead posts (the "viral" incident:
18 shorts titled literally "viral", 0 views).

This module guarantees a usable, clip-specific title line with zero
external calls, in priority order:

    1. the best transcript sentences — scored for hook-ness (questions,
       exclamations, viral/drama lexicon, second person, numbers) — plus a
       cut-off-quote open loop built from the best sentence's first words
    2. a scoreline bank when the stream title parses as "A 1 - 0 B"
       (winner/loser drama; a tie never forces a winner)
    3. a versus bank when the title parses as "A vs B" (never a result claim)
    4. a reason-keyed bank dramatizing the detection signal we actually
       measured (audio roar / chat spike / speech / interval)
    5. a language-appropriate generic template, picked stably from the seed

The raw or cleaned stream title never ships verbatim (a wall of identical
"Mexico 1 - 0 Korea" posts reads as spam to viewers and ranking systems),
and truth discipline mirrors the LLM prompt: templates dramatize only what
the inputs support — a parsed score, a matchup, a measured signal — and
never invent repetition, results, or accusations.

Pure functions, no I/O, no randomness beyond crc32(seed). The LLM path
stays primary everywhere — these are the fallback the publish flow can
always count on.
"""

from __future__ import annotations

import re
import zlib

MAX_HOOK_WORDS = 12

# A hook burns onto the video as a single line — a hard budget, not a style
# preference. Bank fills that blow it are skipped, never truncated.
_MAX_HOOK_CHARS = 90

# Sentence boundary: ., !, ?, … followed by whitespace/end. Keeps the
# terminal punctuation attached to the sentence.
_SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]*")

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#\w+")
_WS_RE = re.compile(r"\s+")

# Openers that mark a sentence as filler/ramble rather than a hook.
_FILLER_STARTS = frozenset(
    {
        # es
        "eh", "em", "este", "bueno", "entonces", "pues", "osea", "o", "ya",
        "aja", "ajá", "mmm", "vale", "digo",
        # en
        "um", "uh", "so", "like", "okay", "ok", "well", "anyway", "yeah",
    }
)

# Words that signal a charged / curiosity-driving moment (both languages,
# lowercase, accent-insensitive matching is NOT attempted — common forms
# are listed explicitly).
_VIRAL_LEXICON = frozenset(
    {
        # es
        "increible", "increíble", "locura", "loco", "nunca", "jamas", "jamás",
        "nadie", "todos", "secreto", "dinero", "gratis", "error", "verdad",
        "mentira", "peor", "mejor", "ultimo", "último", "final", "gano",
        "ganó", "perdio", "perdió", "explota", "insulto", "record", "récord",
        "prohibido", "viral",
        # en
        "insane", "crazy", "never", "nobody", "everyone", "secret", "money",
        "free", "mistake", "truth", "lie", "worst", "best", "last", "finally",
        "won", "lost", "exposed", "banned", "unhinged", "wild",
    }
)

# Multi-word charged phrases, matched as substrings of the lowered sentence
# (they can't live in the word-level lexicon). Their presence marks the
# sentence as the kind of speech clips are made of.
_DRAMA_LEX = frozenset(
    {
        # es
        "se pasó", "no puedo creer", "otra vez", "en mi cara", "se vino abajo",
        # en
        "can't believe", "no way",
    }
)

_SECOND_PERSON = frozenset(
    {"tú", "tu", "te", "usted", "ustedes", "you", "your", "u"}
)

# Stream-title junk the topic extractor strips. The timestamp group is also
# re-run on every FILLED template — a date/time/resolution fragment never
# ships even if it survives extraction. ISO dates go before bare years so
# "2026-07-15" doesn't decay into "-07-15".
_TIMESTAMP_JUNK_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),  # ISO dates
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),  # slash dates
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),  # clock times
    re.compile(r"\b(?:19|20)\d{2}\b"),  # bare years
    re.compile(r"\b(?:\d{3,4}p|4k|\d{2,3}\s*fps)\b", re.IGNORECASE),  # res/fps
)
_NOISE_JUNK_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:vod|directo|stream|live|gameplay|parte\s*\d+|ep\.?\s*\d+|d[ií]a\s*\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\[[^\]]*\]"),  # [bracket tags]
)

# "Mexico 1 - 0 Korea" — sides may not contain digits or pipes; the score
# separator tolerates spaces ("1 - 0"), en dashes, and colons.
_SCORE_RE = re.compile(
    r"([^\d|]{2,30}?)\s*(\d{1,2})\s*[-\u2013:]\s*(\d{1,2})\s*([^\d|]{2,30})"
)
_VERSUS_RE = re.compile(r"(.{2,30}?)\s+(?:vs\.?|contra)\s+(.{2,30})", re.IGNORECASE)
# Dividers (and a nested vs/contra) that bound a team name inside a matched
# side: "World Cup - England" -> "England".
_SIDE_SPLIT_RE = re.compile(r"[-\u2013—:|,]|\bvs\.?\b|\bcontra\b", re.IGNORECASE)

# Template banks. Truth discipline: score templates fire only on a parsed
# scoreline and claim nothing the scoreline doesn't support (no invented
# repetition, no accusations of cheating/lying/crimes); versus templates
# never claim a result; reason templates dramatize the detection signal we
# actually measured. {topic} slots need a >=2-word topic to fill.
_SCORE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "es": (
        "El momento en que {winner} rompió el partido ⚽",
        "Así se le escapó el partido a {loser} 💔",
        "El {sa}-{sb} que dejó helado a {loser} 😱",
        "El {sa}-{sb} de {winner} que hizo explotar el directo 🔥",
    ),
    "en": (
        "The moment {winner} broke the game open ⚽",
        "How the game slipped away from {loser} 💔",
        "The {sa}-{sb} that left {loser} frozen 😱",
        "{winner}'s {sa}-{sb} had the whole stream screaming 🔥",
    ),
}
_TIE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "es": (
        "El empate que nadie pidió 😭",
        "Ni {a} ni {b}… pero QUÉ momento",
    ),
    "en": (
        "The draw nobody asked for 😭",
        "Neither {a} nor {b}… but WHAT a moment",
    ),
}
_VERSUS_TEMPLATES: dict[str, tuple[str, ...]] = {
    "es": (
        "{a} vs {b}: el momento que hizo gritar a TODOS",
        "Lo que pasó en {a} vs {b} no fue normal 😱",
        "{a} vs {b} y el directo se vino ABAJO",
        "El instante de {a} vs {b} que vas a ver dos veces",
    ),
    "en": (
        "{a} vs {b}: the moment that made EVERYONE scream",
        "What happened in {a} vs {b} was not normal 😱",
        "{a} vs {b} and the stream went WILD",
        "The {a} vs {b} moment you'll watch twice",
    ),
}
_REASON_TEMPLATES: dict[tuple[str, str], tuple[str, ...]] = {
    ("audio", "es"): (
        "El grito que rompió el directo 🔥",
        "Sube el volumen… esto se escuchó FUERTE 🔊",
        "El momento de {topic} que hizo explotar a todos",
        "Este rugido no era normal 😱",
        "Se escuchó hasta en el chat 💀",
    ),
    ("audio", "en"): (
        "The roar that broke the stream 🔥",
        "Turn the volume up… this got LOUD 🔊",
        "The {topic} moment that blew everyone away",
        "That roar was not normal 😱",
        "You could hear it all over the chat 💀",
    ),
    ("chat", "es"): (
        "El chat se volvió LOCO con este momento 💀",
        "Cuando el chat explota así, es por algo…",
        "El momento de {topic} que reventó el chat",
        "Todo el chat escribiendo a la vez… mira por qué",
    ),
    ("chat", "en"): (
        "The chat went CRAZY at this moment 💀",
        "When the chat blows up like this, there's a reason…",
        "The {topic} moment that broke the chat",
        "The whole chat typing at once… see why",
    ),
    ("voice", "es"): (
        "Lo dijo EN DIRECTO y sin filtro 😳",
        "La frase que nadie vio venir",
        "No se guardó NADA 😭",
        "Lo que soltó aquí dejó a todos así 💀",
    ),
    ("voice", "en"): (
        "Said it LIVE with zero filter 😳",
        "The line nobody saw coming",
        "Held NOTHING back 😭",
        "What got said here left everyone like this 💀",
    ),
    ("interval", "es"): (
        "El momento de {topic} que te perdiste",
        "Esto pasó en {topic} y nadie lo comenta",
        "Espera al final de este momento…",
        "De los directos salen estas joyas 💎",
    ),
    ("interval", "en"): (
        "The {topic} moment you missed",
        "This happened in {topic} and nobody talks about it",
        "Wait for the end of this one…",
        "Streams keep dropping gems like this 💎",
    ),
}

# Last-resort templates. Picked by a stable hash of the seed, so two clips
# from the same batch land on different lines — a wall of identical titles
# reads as spam to both viewers and ranking systems.
_TEMPLATES: dict[str, tuple[str, ...]] = {
    "es": (
        "El momento que nadie vio venir",
        "Esto se salió de control en segundos",
        "Nadie esperaba lo que pasó aquí",
        "Espera al final — no es lo que crees",
        "Esto pasó en vivo y quedó grabado",
        "La reacción que todos están comentando",
    ),
    "en": (
        "The moment nobody saw coming",
        "This went off the rails in seconds",
        "Nobody expected what happened here",
        "Wait for the end — not what you think",
        "This happened live and it's all on tape",
        "The reaction everyone is talking about",
    ),
}


def _lang_key(language: str) -> str:
    return "es" if (language or "").strip().lower().startswith("es") else "en"


def _norm(text: str) -> str:
    """Comparison key: casefolded, whitespace-collapsed."""
    return " ".join((text or "").casefold().split())


def _norm_reason(reason: str) -> str:
    """Collapse a detect reason ("voice trigger", "chat", "audio peak")
    onto a bank key by substring; anything unrecognized reads as interval."""
    lowered = (reason or "").lower()
    for key in ("voice", "chat", "audio"):
        if key in lowered:
            return key
    return "interval"


def _words(text: str) -> list[str]:
    return [w for w in _WS_RE.split(text.strip()) if w]


def _split_sentences(text: str) -> list[str]:
    return [m.group(0).strip() for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]


def _score_sentence(sentence: str) -> int:
    """Hook-ness score for one transcript sentence. Higher = better."""
    words = _words(sentence)
    if not words:
        return -10
    n = len(words)
    score = 0
    if 5 <= n <= 14:
        score += 2
    elif 3 <= n <= 18:
        score += 1
    else:
        score -= 1
    if "?" in sentence:
        score += 2
    if "!" in sentence:
        score += 2
    lowered = [w.strip(".,!?…\"'()").lower() for w in words]
    lexicon_hits = sum(1 for w in lowered if w in _VIRAL_LEXICON)
    score += min(lexicon_hits, 2) * 2
    # Charged phrases ("se pasó", "no puedo creer") beat flat recap.
    if any(phrase in sentence.casefold() for phrase in _DRAMA_LEX):
        score += 2
    if any(w in _SECOND_PERSON for w in lowered):
        score += 1
    if any(any(ch.isdigit() for ch in w) for w in words):
        score += 1
    if lowered[0] in _FILLER_STARTS:
        score -= 2
    return score


def _trim_to_hook(sentence: str, *, max_words: int = MAX_HOOK_WORDS) -> str:
    """Normalize a sentence into title shape: collapsed whitespace, capped
    word count (truncation keeps an ellipsis as an open loop), no trailing
    period, capitalized first letter."""
    words = _words(sentence)
    truncated = len(words) > max_words
    text = " ".join(words[:max_words]).strip().rstrip(",;:—-")
    if truncated:
        text = text.rstrip(".!?…") + "…"
    text = text.rstrip(".")
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _cutoff_quote(sentence: str) -> str:
    """Open-loop fragment from real speech: the first ~5 words of the best
    scored sentence as «frag»… — a curiosity gap the viewer resolves by
    watching. Only built from an actual sentence, never a template."""
    frag = " ".join(_words(sentence)[:5]).strip(" .,;:!?¡¿…\"'«»")
    if frag and frag[0].islower():
        frag = frag[0].upper() + frag[1:]
    return f"«{frag}»…" if frag else ""


def clean_stream_title(title: str, *, max_words: int = MAX_HOOK_WORDS) -> str:
    """Strip stream-title noise (URLs, hashtags, chatbot commands, pipe
    segments) down to the human part, trimmed to hook length."""
    text = _URL_RE.sub("", title or "")
    text = _HASHTAG_RE.sub("", text)
    # "!discord"-style chatbot commands are noise in a title.
    text = re.sub(r"(?<!\w)![a-z]\w*", "", text, flags=re.IGNORECASE)
    # Streamers chain segments with | or •; the longest one is the story.
    segments = [s.strip() for s in re.split(r"[|•]", text) if s.strip()]
    if not segments:
        return ""
    best = max(segments, key=lambda s: len(_words(s)))
    return _trim_to_hook(best, max_words=max_words)


def _topic(title: str) -> str:
    """The human topic of a stream title: cleaned, then stripped of dates,
    clock times, years, resolution junk, noise words, and bracket tags —
    "World Cup - England vs Argentina 2026-07-15 18:31" becomes
    "World Cup - England vs Argentina"."""
    text = clean_stream_title(title)
    for rx in (*_TIMESTAMP_JUNK_RES, *_NOISE_JUNK_RES):
        text = rx.sub(" ", text)
    return _WS_RE.sub(" ", text).strip(" -\u2013—:|…")


def _topic_slot(topic: str) -> str:
    """Topic as slotted into a {topic} template: only tokens carrying
    alphanumerics (drops divider shrapnel like a lone "-"), capped at 5
    words so the fill stays inside the char budget."""
    words = [w for w in _words(topic) if any(ch.isalnum() for ch in w)]
    return " ".join(words[:5])


def _side(raw: str, *, last: bool) -> str:
    """Trim one matched side to the team name nearest the separator: the
    last <=3 words of the left side, the first <=3 of the right."""
    parts = [p.strip() for p in _SIDE_SPLIT_RE.split(raw) if p and p.strip()]
    if not parts:
        return ""
    words = _words(parts[-1] if last else parts[0])
    return " ".join(words[-3:] if last else words[:3])


def _parse_score(topic: str) -> tuple[str, int, int, str] | None:
    """Parse "Mexico 1 - 0 Korea"-shaped topics -> (team_a, sa, sb, team_b)."""
    m = _SCORE_RE.search(topic)
    if m is None:
        return None
    team_a = _side(m.group(1), last=True)
    team_b = _side(m.group(4), last=False)
    if not team_a or not team_b:
        return None
    return team_a, int(m.group(2)), int(m.group(3)), team_b


def _parse_versus(topic: str) -> tuple[str, str] | None:
    """Parse "… England vs Argentina …"-shaped topics -> (a, b)."""
    m = _VERSUS_RE.search(topic)
    if m is None:
        return None
    a = _side(m.group(1), last=True)
    b = _side(m.group(2), last=False)
    if not a or not b:
        return None
    return a, b


def _fill(template: str, fields: dict[str, str]) -> str:
    """Fill one bank template. Returns "" when the fill is unusable — a
    missing field, an unfilled brace, or a date/time/resolution fragment
    that survived into the FILLED string (junk guards re-run post-fill)."""
    try:
        text = template.format(**fields)
    except (KeyError, IndexError):
        return ""
    if "{" in text or "}" in text:
        return ""
    if any(rx.search(text) for rx in _TIMESTAMP_JUNK_RES):
        return ""
    return _WS_RE.sub(" ", text).strip()


def deterministic_hook(
    *,
    transcript_snippet: str = "",
    stream_title: str = "",
    language: str = "es",
    reason: str = "",
    platform: str = "",
    seed: str = "",
    max_words: int = MAX_HOOK_WORDS,
    avoid: set[str] | frozenset[str] = frozenset(),
) -> str:
    """One usable title line, guaranteed non-empty, with zero LLM calls.

    `seed` should be something stable and clip-specific (the clip id) so
    the bank rotation varies across a batch but is idempotent per clip.
    `avoid` carries hooks already used by sibling clips of the same stream
    (compared normalized) so a batch never repeats a title.
    """
    return deterministic_hook_candidates(
        transcript_snippet=transcript_snippet,
        stream_title=stream_title,
        language=language,
        reason=reason,
        platform=platform,
        seed=seed,
        n=1,
        max_words=max_words,
        avoid=avoid,
    )[0]


def deterministic_hook_candidates(
    *,
    transcript_snippet: str = "",
    stream_title: str = "",
    language: str = "es",
    reason: str = "",
    platform: str = "",  # accepted for future per-platform keying; unused in v1
    seed: str = "",
    n: int = 5,
    max_words: int = MAX_HOOK_WORDS,
    avoid: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """Up to `n` distinct title candidates, best first: scored transcript
    sentences (plus a cut-off-quote open loop), then scoreline / versus /
    reason-keyed banks built from the stream title and detection signal,
    then generic templates. Always returns at least one entry."""
    del platform
    n = max(1, n)
    lang = _lang_key(language)
    out: list[str] = []
    seen: set[str] = set()
    avoided = {_norm(a) for a in avoid}
    # The raw or cleaned stream title can never ship verbatim — that was
    # the degenerate wall-of-identical-posts failure mode.
    banned = {
        _norm(stream_title),
        _norm(clean_stream_title(stream_title, max_words=max_words)),
    } - {""}

    def push(text: str) -> None:
        text = text.strip()
        key = _norm(text)
        if not key or key in seen or key in avoided or key in banned:
            return
        if len(text) > _MAX_HOOK_CHARS:  # burns onto the video as one line
            return
        words = _words(text)
        # Two-word "sentences" are transcription shrapnel, not hooks; a
        # candidate with no alphabetic word is bare-timestamp leftovers.
        if len(words) < 3 or not any(any(ch.isalpha() for ch in w) for w in words):
            return
        seen.add(key)
        out.append(text)

    def fill_bank(bank: tuple[str, ...], *, key: str, fields: dict[str, str]) -> None:
        first = zlib.crc32(key.encode("utf-8")) % len(bank)
        for offset in range(len(bank)):
            if len(out) >= n:
                return
            template = bank[(first + offset) % len(bank)]
            # {topic}-slotted templates need a real topic, not one word.
            if "{topic}" in template and len(_words(fields.get("topic", ""))) < 2:
                continue
            text = _fill(template, fields)
            if text:
                push(text)

    # T1 — real speech beats any template: best-scored sentences first,
    # then a cut-off-quote open loop from the single best one. Skipped
    # entirely when there is no scored speech.
    scored = sorted(
        ((s, _score_sentence(s)) for s in _split_sentences(transcript_snippet)),
        key=lambda pair: -pair[1],
    )
    best_speech = ""
    for sentence, score in scored:
        if score >= 2:
            best_speech = best_speech or sentence
            push(_trim_to_hook(sentence, max_words=max_words))
        if len(out) >= n:
            return out[:n]
    if best_speech:
        push(_cutoff_quote(best_speech))

    topic = _topic(stream_title)
    base_fields = {"topic": _topic_slot(topic)}

    # T2 — scoreline bank. A tie downgrades to versus-shape: the tie bank
    # never forces a winner/loser fill.
    if len(out) < n and (parsed := _parse_score(topic)) is not None:
        team_a, sa, sb, team_b = parsed
        if sa == sb:
            fill_bank(
                _TIE_TEMPLATES[lang],
                key=f"{seed}:tie",
                fields={**base_fields, "a": team_a, "b": team_b},
            )
        else:
            winner, loser = (team_a, team_b) if sa > sb else (team_b, team_a)
            fill_bank(
                _SCORE_TEMPLATES[lang],
                key=f"{seed}:score",
                fields={
                    **base_fields,
                    "winner": winner,
                    "loser": loser,
                    "sa": str(sa),
                    "sb": str(sb),
                },
            )

    # T3 — versus bank: matchup drama, never a result claim.
    if len(out) < n and (versus := _parse_versus(topic)) is not None:
        a, b = versus
        fill_bank(
            _VERSUS_TEMPLATES[lang],
            key=f"{seed}:versus",
            fields={**base_fields, "a": a, "b": b},
        )

    # T4 — dramatize the detection signal we actually measured.
    if len(out) < n:
        rkey = _norm_reason(reason)
        fill_bank(_REASON_TEMPLATES[(rkey, lang)], key=f"{seed}:{rkey}", fields=base_fields)

    # T5 — generic pool, last resort (existing rotation, keyed by bare seed).
    template_pool = _TEMPLATES[lang]
    first = zlib.crc32((seed or "").encode("utf-8")) % len(template_pool)
    for offset in range(len(template_pool)):
        if len(out) >= n:
            break
        push(template_pool[(first + offset) % len(template_pool)])

    if not out:
        # An exhaustive avoid set can drain every bank; a repeated title
        # still beats a hookless clip (the "viral" incident failure mode).
        out.append(template_pool[first])
    return out[:n]
