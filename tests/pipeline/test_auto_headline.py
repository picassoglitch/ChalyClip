"""Auto headline (owner, 2026-11-07): a clip's burned top headline is the
hook the model wrote for THAT clip, or nothing. A deterministic template
line ("De los directos salen estas joyas 💎") never becomes the headline,
a clip that already has a headline never gets a second one, and no hook is
generated when the headline is switched off.

Runs the real `process_vod` orchestrator with the SQLite dual-write on, so
the asserted overlay is the row the renderer reads."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from chalybclip.db import ClipsRepo, Database, TenantsRepo, VariantsRepo, apply_migrations
from chalybclip.pipeline import PipelineDeps, _existing_headline, process_vod
from chalybclip.settings import Settings
from chalybclip.tenancy import bound_tenant
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]
from tests.pipeline.test_process_vod import (  # type: ignore[import]
    _force_inprocess_whisper,
    _hook_payload,
    _make_config,
    _make_personas,
    _stub_ffmpeg,
    _stub_ingest,
    _stub_whisper,
)
from tests.transcribe._fakes import (  # type: ignore[import]
    FakeInfo,
    FakeSegment,
    FakeWhisperModel,
    FakeWord,
)


async def _seed(db_path: Path) -> None:
    db = Database(db_path)
    try:
        await apply_migrations(db)
        await TenantsRepo(db).create(tenant_id="default", name="T")
    finally:
        await db.close()


def _patch_provider(monkeypatch: pytest.MonkeyPatch, fake: FakeProvider) -> None:
    from chalybclip.llm import router as router_module

    monkeypatch.setattr(
        router_module, "_default_provider_factory",
        lambda name, _c, _k: fake if name == "anthropic" else None,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeProvider,
    *,
    settings: Settings | None = None,
    stream_id: str | None = None,
    stub_whisper: Any = _stub_whisper,
) -> Any:
    db_path = tmp_path / "t.db"
    if not db_path.exists():
        asyncio.run(_seed(db_path))
    _stub_ingest(monkeypatch)
    stub_whisper(monkeypatch)
    _stub_ffmpeg(monkeypatch)
    _patch_provider(monkeypatch, fake)
    deps = PipelineDeps(
        config=_make_config(),
        llm_config=make_llm_config(retry_attempts=1, purpose="hook_generation"),
        personas=_make_personas(),
        settings=settings,
    )
    return asyncio.run(
        process_vod(
            tenant_id="default",
            vod_url="https://kick.com/aldovillanueva/videos/abc",
            output_dir=tmp_path,
            persona_id="aldo_villanueva",
            language="es",
            n_variants=1,
            db_path=str(db_path),
            stream_id=stream_id,
            deps=deps,
        )
    )


def _clip_rows(tmp_path: Path, stream_id: str) -> list[Any]:
    async def _go() -> list[Any]:
        db = Database(tmp_path / "t.db")
        try:
            with bound_tenant("default"):
                return await ClipsRepo(db).list_for_stream(stream_id)
        finally:
            await db.close()

    return asyncio.run(_go())


def _headlines(overlay: dict[str, Any] | None) -> list[str]:
    """Every header the renderer would burn for this overlay."""
    ov = overlay or {}
    out: list[str] = []
    if (ov.get("title_text") or "").strip():
        out.append(ov["title_text"])
    top = ov.get("top_hook") or {}
    if top.get("enabled") and (top.get("text") or "").strip():
        out.append(top["text"])
    return out


def test_model_hook_becomes_the_one_headline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeProvider("anthropic")
    fake.queue_success(_hook_payload())
    manifest = _run(tmp_path, monkeypatch, fake)
    [row] = _clip_rows(tmp_path, manifest.stream.id)
    assert _headlines(row.overlay_config) == ["HOOK VIRAL"]
    assert len(fake.calls) == 1


def test_no_model_hook_means_no_headline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model down → the clip keeps a deterministic caption title, but the
    video carries no generic banner."""
    fake = FakeProvider("anthropic")
    fake.queue_fatal("llm down")
    manifest = _run(tmp_path, monkeypatch, fake)
    [row] = _clip_rows(tmp_path, manifest.stream.id)
    assert _headlines(row.overlay_config) == []
    assert "top_hook" not in (row.overlay_config or {})
    # The rest of the viral default still ships (captions + source credit).
    assert row.overlay_config["captions"]["enabled"] is True
    assert row.overlay_config["banner"]["url"] == "aldovillanueva"
    # The caption title still falls back to a deterministic line.
    assert manifest.clip_entries[0].variants[0].title_card_text.strip()
    cache = json.loads(
        (manifest.clip_entries[0].clip.path.parent / "variants.json").read_text("utf-8")
    )
    assert cache["headline"] is False


def test_headline_disabled_spends_no_llm_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeProvider("anthropic")
    fake.queue_success(_hook_payload())
    manifest = _run(tmp_path, monkeypatch, fake, settings=Settings(auto_hook_enabled=False))
    [row] = _clip_rows(tmp_path, manifest.stream.id)
    assert fake.calls == []
    assert _headlines(row.overlay_config) == []


def test_existing_headline_is_kept_and_not_doubled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume after the operator set a title: the variants step reruns but
    keeps that one header and pays for no new hook."""
    fake = FakeProvider("anthropic")
    fake.queue_success(_hook_payload())
    fake.queue_success({"hooks": [{"text": "OTRO HOOK"}]})
    first = _run(tmp_path, monkeypatch, fake)
    clip = first.clip_entries[0].clip
    edited = {"title_text": "Mi titular", "captions": {"enabled": True}}

    async def _edit() -> None:
        db = Database(tmp_path / "t.db")
        try:
            with bound_tenant("default"):
                await ClipsRepo(db).set_overlay_config(clip.id, overlay_config=edited)
        finally:
            await db.close()

    asyncio.run(_edit())
    (clip.path.parent / "variants.json").unlink()  # variants step reruns

    second = _run(tmp_path, monkeypatch, fake, stream_id=first.stream.id)
    [row] = _clip_rows(tmp_path, second.stream.id)
    assert row.overlay_config == edited
    assert _headlines(row.overlay_config) == ["Mi titular"]
    assert len(fake.calls) == 1  # no hook bought for a headline already set
    assert second.clip_entries[0].variants[0].title_card_text == "Mi titular"

    async def _variants() -> list[Any]:
        db = Database(tmp_path / "t.db")
        try:
            with bound_tenant("default"):
                return await VariantsRepo(db).list_for_clip(clip.id)
        finally:
            await db.close()

    assert [v.title_card_text for v in asyncio.run(_variants())] == ["Mi titular"]


def test_overlay_read_failure_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex #106 P2: if the overlay read fails, the stored overlay (custom
    settings, no headline) is left as is and no hook is bought."""
    import chalybclip.pipeline as pipeline_mod

    fake = FakeProvider("anthropic")
    fake.queue_success(_hook_payload())
    fake.queue_success({"hooks": [{"text": "OTRO HOOK"}]})
    first = _run(tmp_path, monkeypatch, fake)
    clip = first.clip_entries[0].clip
    custom = {"captions": {"enabled": False}, "banner": {"enabled": True, "url": "mio"}}

    async def _edit() -> None:
        db = Database(tmp_path / "t.db")
        try:
            with bound_tenant("default"):
                await ClipsRepo(db).set_overlay_config(clip.id, overlay_config=custom)
        finally:
            await db.close()

    asyncio.run(_edit())
    (clip.path.parent / "variants.json").unlink()

    class _FlakyRead(ClipsRepo):
        """The pipeline's read blips; the write path (incl. its own lookup) works."""

        _writing = False

        async def get(self, clip_id: str) -> Any:
            if not self._writing:
                raise RuntimeError("db blip")
            return await super().get(clip_id)

        async def set_overlay_config(self, clip_id: str, **kw: Any) -> Any:
            self._writing = True
            return await super().set_overlay_config(clip_id, **kw)

    async def _no_auto_correct(*_a: Any, **_k: Any) -> dict[str, object]:
        return {}  # isolate the variants step from the auto-correct pass

    import chalybclip.clip as clip_pkg

    monkeypatch.setattr(clip_pkg, "auto_correct_clip", _no_auto_correct)
    monkeypatch.setattr(pipeline_mod, "ClipsRepo", _FlakyRead)
    second = _run(tmp_path, monkeypatch, fake, stream_id=first.stream.id)
    monkeypatch.undo()
    [row] = _clip_rows(tmp_path, second.stream.id)
    assert row.overlay_config == custom
    assert len(fake.calls) == 1  # no hook bought for an overlay we won't write
    assert second.clip_entries[0].variants[0].title_card_text.strip()


@pytest.mark.parametrize(
    ("overlay", "expected"),
    [
        (None, ""),
        ({}, ""),
        ({"title_text": "  T  "}, "T"),
        ({"top_hook": {"enabled": True, "text": "H"}}, "H"),
        ({"top_hook": {"enabled": False, "text": "H"}}, ""),
        ({"top_hook": {"enabled": True, "text": " "}}, ""),
        ({"title_text": "T", "top_hook": {"enabled": True, "text": "H"}}, "T"),
    ],
)
def test_existing_headline_detection(overlay: Any, expected: str) -> None:
    assert _existing_headline(overlay) == expected


def _silent_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    """No trigger words → detect falls back to 5 interval clips."""
    _force_inprocess_whisper(monkeypatch)
    FakeWhisperModel.reset()
    FakeWhisperModel.canned_info = FakeInfo(language="es", duration=60.0)
    FakeWhisperModel.canned_segments = [
        FakeSegment(start=0.0, end=2.0, text="hola",
                    words=[FakeWord(start=0.0, end=2.0, word="hola", probability=0.9)]),
    ]


@pytest.mark.parametrize(("llm_ok", "burned"), [(True, True), (False, False)])
def test_series_parts_inherit_headline_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm_ok: bool, burned: bool
) -> None:
    import chalybclip.pipeline as pipeline_mod

    def one_series(clips: list[Any]) -> dict[str, tuple[int, int, int]]:
        ordered = sorted(clips, key=lambda c: c.start_s)
        return {c.id: (i + 1, len(ordered), 0) for i, c in enumerate(ordered)}

    monkeypatch.setattr(pipeline_mod, "_series_parts", one_series)
    fake = FakeProvider("anthropic")
    if llm_ok:
        fake.queue_success(_hook_payload())
    manifest = _run(tmp_path, monkeypatch, fake, stub_whisper=_silent_whisper)
    rows = _clip_rows(tmp_path, manifest.stream.id)
    assert len(rows) == 5
    assert len(fake.calls) == 1  # one base hook for the whole series
    heads = sorted(h for r in rows for h in _headlines(r.overlay_config))
    if burned:
        assert heads == [f"HOOK VIRAL — Parte {i}/5" for i in range(1, 6)]
    else:
        assert heads == []
    titles = [e.variants[0].title_card_text for e in manifest.clip_entries]
    assert all("Parte" in t for t in titles)  # caption titles still numbered


def test_auto_stamp_is_not_an_existing_headline() -> None:
    from chalybclip.clip.overlay_defaults import default_overlay_config

    stamped = default_overlay_config(platform="kick", channel="a", hook="HOOK")
    assert _existing_headline(stamped) == ""


def test_cached_stub_without_model_hook_burns_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume from variants.json written with headline=false: still no banner."""
    fake = FakeProvider("anthropic")
    fake.queue_fatal("llm down")
    first = _run(tmp_path, monkeypatch, fake)
    second = _run(tmp_path, monkeypatch, fake, stream_id=first.stream.id)
    [row] = _clip_rows(tmp_path, second.stream.id)
    assert _headlines(row.overlay_config) == []
    assert len(fake.calls) == 1
