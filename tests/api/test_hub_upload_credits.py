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
FREE_CREDITS = 50_000  # TIER_CAPS.FREE.tokensPerMonth
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
                    "remaining": remaining,
                    "needed": needed,
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


async def _free_tenant(db: Database, tenant_id: str) -> dict[str, str]:
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
        tier="free",
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
    # The QA failure, reproduced: no length → 10-min estimate → refused,
    # and the refusal now says how much it takes and how much there is.
    hdr = await _free_tenant(db, tenants["alice"]["id"])
    r = await client.post("/api/hub/v1/uploads", json=_upload(None), headers=hdr)
    assert r.status_code == 402, r.text
    err = r.json()["error"]
    assert err["code"] == "no_credits" and err["reason"] == "no_tokens"
    needed = math.ceil(estimate_tokens(source_minutes=0) * 2.6)
    assert needed > FREE_CREDITS
    assert err["credits"] == {"needed": needed, "remaining": FREE_CREDITS}


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
    # /complete once ffprobe measures it.
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
    assert r.status_code == 402, r.text
    assert r.json()["error"]["reason"] == "no_tokens"
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
