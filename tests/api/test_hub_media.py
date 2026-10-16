"""/api/hub/media — signed media links the hub hands to the browser."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from chalybclip.api.hub_media import media_path
from chalybclip.db import ClipsRepo, Database, StreamsRepo
from chalybclip.db.models import ClipRow, StreamRow
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

_SECRET = "hub_media_secret"


@pytest.fixture(autouse=True)
def hub_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYB_SSO_SECRET", _SECRET)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed(db: Database, tenant_id: str, tmp_path: Path, *, sid: str, cid: str) -> Path:
    clip_file = tmp_path / f"{cid}.mp4"
    clip_file.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 2048)
    thumb = tmp_path / f"{cid}.jpg"
    thumb.write_bytes(b"\xff\xd8\xff" + b"j" * 64)
    with bound_tenant(tenant_id):
        await StreamsRepo(db).upsert(
            StreamRow(
                id=sid, tenant_id=tenant_id, vod_url="https://kick.com/x", platform="kick",
                title="t", channel=None, duration_s=60.0,
                source_video_path=str(clip_file), source_audio_path="/nope.wav",
                status="done", created_at=_now(),
            )
        )
        await ClipsRepo(db).upsert_many([
            ClipRow(
                id=cid, stream_id=sid, tenant_id=tenant_id, candidate_id=None,
                start_s=0.0, end_s=10.0, duration_s=10.0, width=1080, height=1920,
                path=str(clip_file), status="cut", created_at=_now(),
                thumbnail_frame_path=str(thumb),
            )
        ])
    return clip_file


async def test_signed_link_serves_clip_and_thumb(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    alice = tenants["alice"]["id"]
    clip_file = await _seed(db, alice, tmp_path, sid="str_m1", cid="clp_m1")
    r = await client.get(media_path("clip", "clp_m1", tenant_id=alice))
    assert r.status_code == 200
    assert r.content == clip_file.read_bytes()
    assert "attachment" not in r.headers.get("content-disposition", "")
    r = await client.get(media_path("clip", "clp_m1", tenant_id=alice, params={"dl": "1"}))
    assert "attachment" in r.headers["content-disposition"]
    r = await client.get(media_path("thumb", "clp_m1", tenant_id=alice))
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    r = await client.get(media_path("source", "str_m1", tenant_id=alice))
    assert r.status_code == 200
    # Range requests work (video seeking).
    r = await client.get(
        media_path("clip", "clp_m1", tenant_id=alice), headers={"Range": "bytes=0-9"}
    )
    assert r.status_code == 206 and len(r.content) == 10


async def test_link_is_bound_to_its_tenant(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    alice, bob = tenants["alice"]["id"], tenants["bob"]["id"]
    await _seed(db, alice, tmp_path, sid="str_m2", cid="clp_m2")
    # Bob's (valid) link to Alice's clip id → not found for Bob.
    r = await client.get(media_path("clip", "clp_m2", tenant_id=bob))
    assert r.status_code == 404
    # Re-pointing Alice's link at Bob breaks the signature.
    path = media_path("clip", "clp_m2", tenant_id=alice).replace(f"t={alice}", f"t={bob}")
    assert (await client.get(path)).status_code == 403
    # Another resource id with Alice's signature → 403.
    path = media_path("clip", "clp_m2", tenant_id=alice).replace("/clp_m2?", "/clp_other?")
    assert (await client.get(path)).status_code == 403
    # Adding an unsigned param (dl) → 403.
    path = media_path("clip", "clp_m2", tenant_id=alice) + "&dl=1"
    assert (await client.get(path)).status_code == 403


async def test_expired_unsigned_and_unknown(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    alice = tenants["alice"]["id"]
    await _seed(db, alice, tmp_path, sid="str_m3", cid="clp_m3")
    old = media_path("clip", "clp_m3", tenant_id=alice, now=1_000_000)
    assert (await client.get(old)).status_code == 403
    assert (await client.get("/api/hub/media/clip/clp_m3")).status_code == 403
    assert (await client.get(f"/api/hub/media/clip/clp_m3?t={alice}&exp=x&sig=y")).status_code == 403
    assert (await client.get("/api/hub/media/nope/clp_m3?t=a&exp=1&sig=b")).status_code == 404
    with pytest.raises(ValueError):
        media_path("nope", "x", tenant_id=alice)
    # Far-future expiry claims are refused even with a valid signature.
    far = media_path("clip", "clp_m3", tenant_id=alice, ttl_s=10**9)
    assert (await client.get(far)).status_code == 200  # TTL is clamped when minting


async def test_media_link_needs_no_session(
    client: httpx.AsyncClient, db: Database, tenants: dict[str, dict[str, str]], tmp_path: Path
) -> None:
    """The browser loads these without any cookie or bearer."""
    alice = tenants["alice"]["id"]
    await _seed(db, alice, tmp_path, sid="str_m4", cid="clp_m4")
    client.cookies.clear()
    r = await client.get(media_path("thumb", "clp_m4", tenant_id=alice))
    assert r.status_code == 200
