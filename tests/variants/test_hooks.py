"""Hook generator service — slice F.7-B."""

from __future__ import annotations

from chalybclip.llm import LLMRouter
from chalybclip.llm.config import ProviderConfig
from chalybclip.variants.hooks import (
    HookBatch,
    _user_prompt,
    generate_hooks,
    tone_choices,
)
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]


def _router_with_hooks(hooks: list[dict[str, str]]) -> tuple[LLMRouter, FakeProvider]:
    """Build a router whose Anthropic provider is a FakeProvider that
    returns the given canned hooks on first call."""
    fake = FakeProvider("anthropic")
    fake.queue_success({"hooks": hooks})

    def factory(name: str, _config: ProviderConfig, _api_key: str) -> object | None:
        return fake if name == "anthropic" else None

    config = make_llm_config(retry_attempts=1, purpose="hook_generation")
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=factory,
    )
    return router, fake


# ---- prompt assembly ----


def test_user_prompt_includes_persona_language_snippet_and_n() -> None:
    p = _user_prompt(
        persona_voice="terse, direct, no emojis",
        persona_language="es",
        transcript_snippet="esto está jodido",
        tone="aggressive",
        n=3,
    )
    assert "terse, direct" in p
    assert "es" in p
    assert "esto está jodido" in p
    assert "Generate 3 title candidates" in p
    # Aggressive tone wording made it through.
    assert "CONFRONTATIONAL" in p


def test_user_prompt_handles_empty_snippet_and_voice() -> None:
    p = _user_prompt(
        persona_voice="",
        persona_language="en",
        transcript_snippet="",
        tone="default",
        n=5,
    )
    # Empty snippet steers the model to the context, not a defeatist
    # "no transcript captured" placeholder that invites meta-commentary.
    assert "transcript unavailable" in p
    assert "no persona voice provided" in p
    assert "Generate 5" in p


def test_user_prompt_surfaces_clip_context() -> None:
    p = _user_prompt(
        persona_voice="x",
        persona_language="es",
        transcript_snippet="",
        tone="default",
        n=3,
        clip_context="Stream title: Mexico 1 - 0 Korea",
    )
    # The stream title reaches the model so a no-speech clip hooks off it.
    assert "Mexico 1 - 0 Korea" in p


def test_user_prompt_unknown_tone_adds_no_extra_instruction() -> None:
    # The clip-culture register in the system prompt IS the default voice;
    # tone presets are operator overrides. Unknown/default tones must not
    # inject an extra instruction line.
    p = _user_prompt(
        persona_voice="x",
        persona_language="en",
        transcript_snippet="y",
        tone="bogus",  # type: ignore[arg-type]
        n=1,
    )
    assert "Extra tone instruction" not in p


def test_user_prompt_non_default_tone_appends_override() -> None:
    p = _user_prompt(
        persona_voice="x",
        persona_language="en",
        transcript_snippet="y",
        tone="curious",
        n=1,
    )
    assert "Extra tone instruction" in p
    assert "OPEN-LOOP" in p


def test_user_prompt_lists_avoid_hooks() -> None:
    p = _user_prompt(
        persona_voice="x",
        persona_language="es",
        transcript_snippet="y",
        tone="default",
        n=1,
        avoid_hooks=("El chat explotó 💀", "  "),
    )
    assert "already used on other clips" in p
    assert "El chat explotó 💀" in p
    # Blank entries are dropped, and no avoid block appears without real ones.
    p2 = _user_prompt(
        persona_voice="x",
        persona_language="es",
        transcript_snippet="y",
        tone="default",
        n=1,
        avoid_hooks=("", "   "),
    )
    assert "already used" not in p2


# ---- generate_hooks (end-to-end via fake provider) ----


async def test_generate_hooks_returns_validated_batch() -> None:
    router, _fake = _router_with_hooks([
        {"text": "Adin Ross asked Clav for RETA"},
        {"text": "Clav fired back fr fr"},
        {"text": "this is why streamers ban each other"},
        {"text": "the fattest moment in Kick history"},
        {"text": "Adin's reaction was UNREAL"},
    ])
    hooks = await generate_hooks(
        tenant_id="default",
        persona_voice="aggressive streamer",
        persona_language="en",
        transcript_snippet="adin called him fat lol",
        tone="aggressive",
        n=5,
        router=router,
    )
    assert len(hooks) == 5
    assert all(h.text for h in hooks)
    assert "Adin Ross" in hooks[0].text


async def test_generate_hooks_clamps_n_to_max() -> None:
    """n=999 must be clamped to MAX_N=10. The fake provider returns
    exactly what it's given; we verify the clamp by checking the
    user-prompt the provider actually saw."""
    router, fake = _router_with_hooks([
        {"text": f"hook {i}"} for i in range(10)
    ])
    await generate_hooks(
        tenant_id="default",
        persona_voice="x",
        persona_language="en",
        transcript_snippet="y",
        n=999,
        router=router,
    )
    # FakeProvider records calls in fake.calls.
    assert "Generate 10 title candidates" in fake.calls[0]["user"]


async def test_generate_hooks_clamps_n_to_min() -> None:
    """n=0 must be clamped UP to MIN_N=1. Without this clamp the
    LLM is asked for 0 hooks and the schema rejects the empty list."""
    router, fake = _router_with_hooks([{"text": "single hook"}])
    await generate_hooks(
        tenant_id="default",
        persona_voice="x",
        persona_language="en",
        transcript_snippet="y",
        n=0,
        router=router,
    )
    assert "Generate 1 title candidates" in fake.calls[0]["user"]


async def test_generate_hooks_passes_purpose_to_router() -> None:
    """Cost tracking depends on every hook generation routing under
    the 'hook_generation' purpose. The router's purpose string must
    flow through to the call log."""
    router, _ = _router_with_hooks([{"text": "x"}])
    await generate_hooks(
        tenant_id="default",
        persona_voice="x",
        persona_language="en",
        transcript_snippet="y",
        router=router,
    )
    # If the purpose wasn't 'hook_generation' the router would have
    # raised a "no routing rule for purpose" error before reaching
    # the fake provider. We got here, so the purpose is correct.


# ---- HookBatch schema ----


def test_hookbatch_rejects_empty_list() -> None:
    """min_length=1 — the LLM returning [] should fail validation."""
    import pytest

    with pytest.raises(Exception):  # noqa: B017 - any pydantic ValidationError
        HookBatch(hooks=[])


def test_hookbatch_rejects_too_long() -> None:
    """max_length=10 — caps runaway responses."""
    import pytest

    with pytest.raises(Exception):  # noqa: B017
        HookBatch(hooks=[{"text": f"h{i}"} for i in range(11)])


def test_hookbatch_rejects_empty_text() -> None:
    """Hook.text has min_length=1 — the LLM must not return empty
    strings as hook candidates."""
    import pytest

    with pytest.raises(Exception):  # noqa: B017
        HookBatch(hooks=[{"text": ""}])


def test_hookbatch_rejects_overly_long_text() -> None:
    """max_length=140 — a hook can't be a paragraph."""
    import pytest

    with pytest.raises(Exception):  # noqa: B017
        HookBatch(hooks=[{"text": "x" * 141}])


# ---- tone_choices public list ----


def test_tone_choices_covers_all_five_presets() -> None:
    ids = {tid for tid, _label in tone_choices()}
    assert ids == {"default", "aggressive", "gen_z", "corporate", "curious"}
