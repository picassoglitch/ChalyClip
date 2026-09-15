"""Deterministic (no-LLM) hook generation — transcript sentence pick,
scoreline/versus/reason template banks, seeded rotation. Pure functions."""

from __future__ import annotations

from chalybclip.variants.deterministic import (
    clean_stream_title,
    deterministic_hook,
    deterministic_hook_candidates,
)


def test_picks_the_most_hooky_transcript_sentence() -> None:
    snippet = (
        "bueno entonces estábamos hablando de eso. "
        "¡No puedo creer que ganó todo el dinero en la última ronda! "
        "y luego seguimos con el stream."
    )
    hook = deterministic_hook(transcript_snippet=snippet, language="es")
    assert "dinero" in hook.lower()
    first_alpha = next(c for c in hook if c.isalpha())
    assert first_alpha.isupper()


def test_questions_and_exclamations_beat_flat_sentences() -> None:
    snippet = (
        "we played some rounds today. "
        "why does nobody ever talk about this trick?"
    )
    hook = deterministic_hook(transcript_snippet=snippet, language="en")
    assert hook.endswith("?")


def test_charged_phrases_outrank_flat_sentences() -> None:
    # "se pasó"-class drama phrases get a scoring bonus over flat recap.
    snippet = (
        "hoy jugamos unas partidas tranquilas con el equipo. "
        "el rival se pasó conmigo en la ranked."
    )
    hook = deterministic_hook(transcript_snippet=snippet, language="es")
    assert "se pasó" in hook.lower()


def test_long_sentences_are_trimmed_with_open_loop() -> None:
    snippet = (
        "esto es increíble porque nunca en toda mi vida había visto una jugada "
        "tan absurda como la que acaba de pasar en esta partida clasificatoria!"
    )
    hook = deterministic_hook(transcript_snippet=snippet, language="es")
    assert len(hook.split()) <= 13  # 12 words + possible trailing ellipsis merge
    assert hook.endswith("…")


def test_charged_speech_yields_a_cutoff_quote_open_loop() -> None:
    snippet = "¡No puedo creer que ganó todo el dinero en la última ronda!"
    candidates = deterministic_hook_candidates(
        transcript_snippet=snippet, language="es", seed="clp_z", n=5,
    )
    # The best sentence's first words become a «frag»… curiosity gap.
    assert any(c.startswith("«") and c.endswith("»…") for c in candidates)


def test_no_speech_means_no_cutoff_quote() -> None:
    candidates = deterministic_hook_candidates(
        stream_title="Mexico 1 - 0 Korea", language="es", seed="clp_z", n=8,
    )
    assert not any(c.startswith("«") for c in candidates)


def test_stream_title_never_ships_verbatim() -> None:
    hook = deterministic_hook(
        transcript_snippet="",
        stream_title="Mexico 1 - 0 Korea | !discord | https://x.com #futbol",
        language="es",
        seed="clp_x",
    )
    assert hook.lower() != "mexico 1 - 0 korea"
    assert "!discord" not in hook
    assert "#futbol" not in hook
    assert "https" not in hook


def test_spaced_scoreline_parses_and_names_the_loser() -> None:
    # "1 - 0" with spaces around the dash must still parse as a scoreline.
    candidates = deterministic_hook_candidates(
        stream_title="Mexico 1 - 0 Korea", language="es", seed="clp_x", n=5,
    )
    joined = " ".join(candidates).lower()
    assert "korea" in joined  # the loser appears inside drama framing
    assert "mexico 1 - 0 korea" not in {c.lower() for c in candidates}


def test_tied_score_never_forces_a_winner() -> None:
    candidates = deterministic_hook_candidates(
        stream_title="Mexico 2 - 2 Korea", language="es", seed="clp_t", n=5,
    )
    joined = " ".join(candidates).lower()
    assert "empate" in joined or "ni mexico ni korea" in joined
    # No winner/loser drama was fabricated for a draw.
    assert "rompió el partido" not in joined
    assert "se le escapó" not in joined


def test_junky_sports_title_never_leaks_timestamps_or_the_title() -> None:
    title = "World Cup - England vs Argentina 2026-07-15 18:31"
    candidates = deterministic_hook_candidates(
        stream_title=title, language="es", reason="audio peak", seed="clp_a", n=5,
    )
    cleaned = clean_stream_title(title).lower()
    for c in candidates:
        low = c.lower()
        assert "2026" not in low
        assert "07-15" not in low
        assert "18:31" not in low
        assert low != cleaned
        assert low != title.lower()
    # Different seeds start the bank rotation at different offsets.
    hooks = {
        deterministic_hook(
            stream_title=title, language="es", reason="audio peak", seed=f"clp_{i}"
        )
        for i in range(8)
    }
    assert len(hooks) > 1


def test_one_word_topic_never_leaks_unfilled_braces() -> None:
    # A 1-word topic makes {topic}-slotted templates ineligible; whatever
    # fills instead must never carry an unfilled brace.
    candidates = deterministic_hook_candidates(
        stream_title="Valorant", language="es", reason="audio peak", seed="clp_q", n=8,
    )
    assert candidates
    assert all("{" not in c and "}" not in c for c in candidates)


def test_avoid_set_excludes_already_used_hooks() -> None:
    first = deterministic_hook(language="es", seed="clp_a")
    assert deterministic_hook(language="es", seed="clp_a", avoid={first}) != first
    # Normalized compare: case and stray whitespace don't dodge the filter.
    shouty = "  " + first.upper() + "  "
    assert deterministic_hook(language="es", seed="clp_a", avoid={shouty}) != first
    candidates = deterministic_hook_candidates(
        language="es", seed="clp_a", n=5, avoid=frozenset({first})
    )
    assert first not in candidates


def test_template_fallback_is_stable_per_seed_and_varies_across_seeds() -> None:
    a1 = deterministic_hook(language="es", seed="clp_a")
    a2 = deterministic_hook(language="es", seed="clp_a")
    assert a1 == a2  # idempotent per clip
    hooks = {deterministic_hook(language="es", seed=f"clp_{i}") for i in range(12)}
    assert len(hooks) > 1  # a batch never ships one identical wall of titles


def test_reason_routes_the_template_bank() -> None:
    chat = deterministic_hook(language="es", reason="chat", seed="x")
    audio = deterministic_hook(language="es", reason="audio", seed="x")
    assert chat != audio


def test_hook_is_never_empty() -> None:
    assert deterministic_hook() != ""
    assert deterministic_hook(transcript_snippet="eh", stream_title="", seed="") != ""


def test_language_routes_template_pool() -> None:
    es = deterministic_hook(language="es", seed="x")
    en = deterministic_hook(language="en", seed="x")
    assert es != en


def test_candidates_are_distinct_capped_and_within_budget() -> None:
    snippet = (
        "¡Esto fue una locura total! "
        "¿Cómo es posible que nadie lo viera venir? "
        "Nunca había pasado algo así en el canal."
    )
    candidates = deterministic_hook_candidates(
        transcript_snippet=snippet, stream_title="Gran final del torneo",
        language="es", seed="clp_x", n=5,
    )
    assert 1 <= len(candidates) <= 5
    assert len({c.lower() for c in candidates}) == len(candidates)
    assert all(len(c) <= 90 for c in candidates)


def test_clean_stream_title_prefers_longest_segment() -> None:
    assert clean_stream_title("!prime | GRAN FINAL contra el campeón | !sub") == (
        "GRAN FINAL contra el campeón"
    )
