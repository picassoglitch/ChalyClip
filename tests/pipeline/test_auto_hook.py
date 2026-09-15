"""Pipeline auto-hook: every publish-worthy clip gets a viral title line
generated from its transcript, best-effort (never breaks the run)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chalybclip.pipeline import _auto_hook_for_clip
from chalybclip.variants.hooks import Hook, HookBatch


class _FakeRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kw: Any) -> HookBatch:
        self.calls.append(kw)
        assert kw["purpose"] == "hook_generation"
        return HookBatch(hooks=[Hook(text="Esto Cambió Toda La Partida")])


def _clip(snippet: str) -> Any:
    return SimpleNamespace(
        id="clp_x", candidate=SimpleNamespace(evidence={"transcript_snippet": snippet})
    )


_PERSONA = SimpleNamespace(voice_prompt="energético, directo", primary_language="es")


@pytest.mark.asyncio
async def test_generates_hook_from_transcript_snippet() -> None:
    router = _FakeRouter()
    hook = await _auto_hook_for_clip(
        clip=_clip("clipea esto, esa jugada fue una locura"),
        persona=_PERSONA, language="es", tenant_id="t1", router=router,
    )
    assert hook == "Esto Cambió Toda La Partida"
    # The clip's transcript snippet was passed through to the generator.
    assert "clipea esto" in router.calls[0]["user"]


class _Boom:
    async def complete(self, **kw: Any) -> HookBatch:
        raise RuntimeError("llm down")


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_deterministic_hook() -> None:
    """A hook hiccup never breaks the pipeline AND never ships a hookless
    clip — the deterministic generator takes over (the "viral" incident:
    empty hooks cascaded into one-word captions on 18 dead posts)."""
    hook = await _auto_hook_for_clip(
        clip=_clip("¡esa jugada fue una locura y nadie la vio venir!"),
        persona=_PERSONA, language=None, tenant_id="t1", router=_Boom(),
    )
    assert hook != ""
    assert "locura" in hook.lower()  # built from the clip's own transcript


@pytest.mark.asyncio
async def test_llm_failure_with_no_transcript_hooks_off_the_stream_title() -> None:
    hook = await _auto_hook_for_clip(
        clip=_clip(""), persona=_PERSONA, language=None,
        tenant_id="t1", router=_Boom(), stream_title="Mexico 1 - 0 Korea",
    )
    # The scoreline is parsed into drama framing — never the title verbatim
    # (the old wall-of-identical-titles failure mode).
    assert hook != "Mexico 1 - 0 Korea"
    assert "mexico" in hook.lower() or "korea" in hook.lower()


@pytest.mark.asyncio
async def test_llm_failure_with_nothing_at_all_still_returns_a_title() -> None:
    hook = await _auto_hook_for_clip(
        clip=_clip(""), persona=_PERSONA, language=None,
        tenant_id="t1", router=_Boom(),
    )
    assert hook != ""  # template fallback — a clip never ships hookless
