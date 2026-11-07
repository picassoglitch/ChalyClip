"""QA 2026-11-06: a Free account with 50,000 credits was refused every Clips
upload (100 s and 20 s videos) with "no tienes créditos".

The upload start admitted with an unknown length, which estimate_tokens
counts as 10 minutes; the hub adds its margin, and the hold came out
bigger than the whole Free month. These tests run the real route, the
real admission client and the real estimator against a hub stand-in that
does the hub's own arithmetic (src/lib/usage/admission-core.ts
reserveWithMargin + admit_usage's balance check)."""

from __future__ import annotations

import json as _json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from chalybclip.api.routers import hub_streams
from chalybclip.db import Database, PersonasRepo, TenantsRepo
from chalybclip.integrations.chalyb import admission as admission_mod
from chalybclip.integrations.chalyb.sso import sign_sso_token
from chalybclip.jobs.usage import estimate_tokens
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

_SECRET = "hub_credit_secret"
_HUB = "https://chalyb.test/api/engines/chalybclip/usage"
FREE_CREDITS = 65_000  # TIER_CAPS.FREE.tokensPerMonth (owner 2026-11-07)
HUB_MARGIN_PERCENT = 160  # DEFAULT_USAGE_MARGIN_PERCENT


@pytest.fixture(autouse=True)
def hub_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("CHALYB_SSO_SECRET", _SECRET)
    monkeypatch.setenv("CHALYBCLIP_DEFAULT_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CHALYB_BASE_URL", "https://chalyb.test")
    monkeypatch.setenv("CHALYB_ADMIN_TOKEN", "admintok")
    monkeypatch.setattr(admission_mod, "_ADMIT_RETRY_DELAY_S", 0.0)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def presigned_put_url(self, *, key: str, ttl_seconds: int, content_type: str) -> str:
        return f"https://bucket.test/{key}?sig=put"

    async def presigned_url(self, *, key: str, ttl_seconds: int) -> str:
        return f"https://bucket.test/{key}?sig=get"

    async def size(self, *, key: str) -> int | None:
        return len(self.objects[key]) if key in self.objects else None

    async def download(self, *, key: str, dest: Path) -> Path | None:
        if key not in self.objects:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.objects[key])
        return dest

    async def delete(self, *, key: str) -> None:
        self.objects.pop(key, None)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    import chalybclip.api._pipeline as pipeline_mod
    import chalybclip.integrations.storage as storage_mod

    s = FakeStore()
    monkeypatch.setattr(storage_mod, "build_artifact_store", lambda _settings: s)

    async def fake_runner(**_kw: Any) -> None:
        return None

    monkeypatch.setattr(pipeline_mod, "upload_pipeline_runner", fake_runner)
    return s


class FreeHub:
    """The hub's admit for a Free user holding `balance` credits."""

    def __init__(self, balance: int) -> None:
        self.balance = balance
        self.holds: dict[str, int] = {}
        self.asked: list[dict[str, Any]] = []

    def admit(self, request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content)
        self.asked.append(body)
        needed = math.ceil(body["est_tokens"] * (1 + HUB_MARGIN_PERCENT / 100))
        job = body["external_job_id"]
        remaining = self.balance - sum(v for k, v in self.holds.items() if k != job)
        if remaining < needed:
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "allowed": False,
                    "reason": "no_tokens",
                    "detail": {"remaining": remaining, "needed": needed},
                    "limits": {},
                },
            )
        self.holds[job] = needed
        return httpx.Response(
            200,
            json={
                "ok": True,
                "allowed": True,
                "reservation_id": f"res_{job}",
                "lane": "standard",
                "boost_fee_tokens": 0,
                "limits": {"max_upload_mb": 500},
                "balance": {},
            },
        )


@pytest.fixture
def free_hub() -> Iterator[FreeHub]:
    hub = FreeHub(FREE_CREDITS)
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as mock:
        mock.post(f"{_HUB}/admit").mock(side_effect=hub.admit)
        mock.post(f"{_HUB}/settle").mock(return_value=httpx.Response(200, json={"ok": True}))
        mock.post(f"{_HUB}/events").mock(return_value=httpx.Response(200, json={"ok": True}))
        mock.route(host="api.test").pass_through()
        yield hub


async def _free_tenant(db: Database, tenant_id: str, tier: str = "free") -> dict[str, str]:
    await TenantsRepo(db).set_external_user_id(tenant_id, "free-user-uuid")
    with bound_tenant(tenant_id):
        await PersonasRepo(db).create(
            persona_id="default",
            name="Default",
            primary_language="es",
            target_languages=["es"],
            voice_prompt="x",
            routing_tags=[],
        )
    tok = sign_sso_token(
        user_id="free-user-uuid",
        email="free@example.com",
        tenant_id=tenant_id,
        secret=_SECRET,
        tier=tier,
    )
    return {"Authorization": f"Bearer {tok}"}


def _upload(seconds: float | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "filename": "qa.mp4",
        "size_bytes": 5_000_000,
        "content_type": "video/mp4",
    }
    if seconds is not None:
        body["duration_s"] = seconds
    return body


async def test_unknown_length_hold_is_bigger_than_a_free_month(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
    free_hub: FreeHub,
) -> None:
    # The QA failure, reproduced: no length → 10-min estimate → refused.
    # That figure isn't this video's cost, so the error carries no numbers.
    hdr = await _free_tenant(db, tenants["alice"]["id"])
    r = await client.post("/api/hub/v1/uploads", json=_upload(None), headers=hdr)
    assert r.status_code == 402, r.text
    err = r.json()["error"]
    assert err["code"] == "no_credits" and err["reason"] == "no_tokens"
    assert math.ceil(estimate_tokens(source_minutes=0) * 2.6) > FREE_CREDITS
    assert "credits" not in err


async def test_a_video_too_long_for_the_balance_says_what_it_takes(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
    free_hub: FreeHub,
) -> None:
    # Known length, over what's left this month: the hub's figures (its
    # `detail`, as route.ts answers) reach the error body.
    free_hub.balance = 50_000
    hdr = await _free_tenant(db, tenants["alice"]["id"])
    r = await client.post("/api/hub/v1/uploads", json=_upload(5 * 60), headers=hdr)
    assert r.status_code == 402, r.text
    needed = math.ceil(estimate_tokens(source_minutes=5) * 2.6)
    assert needed > 50_000
    assert r.json()["error"]["credits"] == {"needed": needed, "remaining": 50_000}


@pytest.mark.parametrize("seconds", [20.0, 100.0])
async def test_free_upload_with_its_length_is_admitted_and_runs(
    seconds: float,
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
    free_hub: FreeHub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hdr = await _free_tenant(db, tenants["alice"]["id"])
    r = await client.post("/api/hub/v1/uploads", json=_upload(seconds), headers=hdr)
    assert r.status_code == 200, r.text
    up = r.json()
    first = free_hub.asked[-1]
    assert first["source_minutes"] == round(seconds / 60, 2)
    assert first["est_tokens"] == estimate_tokens(source_minutes=seconds / 60)

    # /complete re-admits the same job with ffprobe's length.
    monkeypatch.setattr(hub_streams, "_probe_minutes_url", lambda _url: seconds / 60)
    key = up["upload_url"].split("bucket.test/")[1].split("?")[0]
    store.objects[key] = b"\x00" * 5_000_000
    r = await client.post(
        f"/api/hub/v1/uploads/{up['stream_id']}/complete",
        json={"ticket": up["ticket"]},
        headers=hdr,
    )
    assert r.status_code == 200, r.text
    again = free_hub.asked[-1]
    assert again["external_job_id"] == first["external_job_id"]
    assert again["source_minutes"] == round(seconds / 60, 2)


async def test_a_short_declared_length_cannot_skip_the_real_check(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
    free_hub: FreeHub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Declares 20 s, is really 10 min: admitted to upload, refused at
    # /complete once ffprobe measures it — Free's 5-minute limit, in its words.
    hdr = await _free_tenant(db, tenants["alice"]["id"])
    up = (await client.post("/api/hub/v1/uploads", json=_upload(20), headers=hdr)).json()
    monkeypatch.setattr(hub_streams, "_probe_minutes_url", lambda _url: 10.0)
    key = up["upload_url"].split("bucket.test/")[1].split("?")[0]
    store.objects[key] = b"\x00" * 5_000_000
    r = await client.post(
        f"/api/hub/v1/uploads/{up['stream_id']}/complete",
        json={"ticket": up["ticket"]},
        headers=hdr,
    )
    assert r.status_code == 413, r.text
    err = r.json()["error"]
    assert err["reason"] == "video_too_long"
    assert err["message"].startswith("Tu plan Gratis permite videos de hasta 5 minutos")
    assert key not in store.objects  # the bytes are dropped


@pytest.mark.parametrize("bad", [0, -1, 24 * 3600 + 1, "veinte"])
async def test_duration_is_validated(
    bad: Any,
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
    free_hub: FreeHub,
) -> None:
    hdr = await _free_tenant(db, tenants["alice"]["id"])
    r = await client.post("/api/hub/v1/uploads", json=_upload(bad), headers=hdr)
    assert r.status_code == 422, r.text
    assert not free_hub.asked


async def test_unreadable_file_is_not_sized_by_the_declared_length(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
    free_hub: FreeHub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ffprobe can't read it → the client's figure is not trusted: the
    # re-admit uses the 10-min estimate, more than Free has.
    hdr = await _free_tenant(db, tenants["alice"]["id"])
    up = (await client.post("/api/hub/v1/uploads", json=_upload(20), headers=hdr)).json()
    monkeypatch.setattr(hub_streams, "_probe_minutes_url", lambda _url: 0.0)
    key = up["upload_url"].split("bucket.test/")[1].split("?")[0]
    store.objects[key] = b"\x00" * 5_000_000
    r = await client.post(
        f"/api/hub/v1/uploads/{up['stream_id']}/complete",
        json={"ticket": up["ticket"]},
        headers=hdr,
    )
    assert r.status_code == 402, r.text
    assert free_hub.asked[-1]["est_tokens"] == estimate_tokens(source_minutes=0)


# ── Free plan: 1 video a month, up to 5 minutes (owner 2026-11-07) ──────


async def _complete(
    client: httpx.AsyncClient,
    store: FakeStore,
    monkeypatch: pytest.MonkeyPatch,
    hdr: dict[str, str],
    up: dict[str, Any],
    minutes: float,
) -> httpx.Response:
    monkeypatch.setattr(hub_streams, "_probe_minutes_url", lambda _url: minutes)
    key = up["upload_url"].split("bucket.test/")[1].split("?")[0]
    store.objects[key] = b"\x00" * 5_000_000
    return await client.post(
        f"/api/hub/v1/uploads/{up['stream_id']}/complete",
        json={"ticket": up["ticket"]},
        headers=hdr,
    )


async def test_free_six_minute_upload_is_refused_before_it_moves(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
    free_hub: FreeHub,
) -> None:
    hdr = await _free_tenant(db, tenants["alice"]["id"])
    r = await client.post("/api/hub/v1/uploads", json=_upload(6 * 60), headers=hdr)
    assert r.status_code == 413, r.text
    err = r.json()["error"]
    assert err["code"] == "too_large" and err["reason"] == "video_too_long"
    assert err["message"] == (
        "Tu plan Gratis permite videos de hasta 5 minutos. Recórtalo o mejora tu plan."
    )
    assert not free_hub.asked  # refused here, no hold, no upload URL
    assert not store.objects


async def test_free_five_minute_video_fits_the_month(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
    free_hub: FreeHub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The month's credits cover a 5-minute video's hold with room to spare.
    hold = math.ceil(estimate_tokens(source_minutes=5) * 2.6)
    assert hold <= FREE_CREDITS
    hdr = await _free_tenant(db, tenants["alice"]["id"])
    r = await client.post("/api/hub/v1/uploads", json=_upload(5 * 60), headers=hdr)
    assert r.status_code == 200, r.text
    r = await _complete(client, store, monkeypatch, hdr, r.json(), 5.0)
    assert r.status_code == 200, r.text
    assert free_hub.asked[-1]["source_minutes"] == 5.0


async def test_free_second_video_in_the_month_is_refused(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
    free_hub: FreeHub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hdr = await _free_tenant(db, tenants["alice"]["id"])
    up = (await client.post("/api/hub/v1/uploads", json=_upload(30), headers=hdr)).json()
    assert (await _complete(client, store, monkeypatch, hdr, up, 0.5)).status_code == 200
    asked = len(free_hub.asked)

    r = await client.post("/api/hub/v1/uploads", json=_upload(30), headers=hdr)
    assert r.status_code == 429, r.text
    err = r.json()["error"]
    assert err["code"] == "rate_limited" and err["reason"] == "videos_cap"
    assert err["message"].startswith("Tu plan Gratis incluye 1 video al mes")
    assert len(free_hub.asked) == asked  # the hub wasn't asked for a second hold


async def test_free_cancelled_upload_does_not_use_the_month(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
    free_hub: FreeHub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hdr = await _free_tenant(db, tenants["alice"]["id"])
    first = (await client.post("/api/hub/v1/uploads", json=_upload(30), headers=hdr)).json()
    r = await client.post(
        f"/api/hub/v1/uploads/{first['stream_id']}/abort",
        json={"ticket": first["ticket"]},
        headers=hdr,
    )
    assert r.status_code == 200, r.text
    free_hub.holds.clear()  # the hub released it on the cancelled settle
    r = await client.post("/api/hub/v1/uploads", json=_upload(30), headers=hdr)
    assert r.status_code == 200, r.text


async def test_pro_long_video_goes_to_the_hub(
    client: httpx.AsyncClient,
    db: Database,
    tenants: dict[str, dict[str, str]],
    store: FakeStore,
    free_hub: FreeHub,
) -> None:
    # The Free limits are Free's: a Pro upload of 6 min is the hub's call.
    free_hub.balance = 10_000_000
    hdr = await _free_tenant(db, tenants["alice"]["id"], tier="pro")
    r = await client.post("/api/hub/v1/uploads", json=_upload(6 * 60), headers=hdr)
    assert r.status_code == 200, r.text
    assert free_hub.asked[-1]["source_minutes"] == 6.0
