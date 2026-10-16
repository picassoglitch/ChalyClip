"""Every account auto-provisioned on SSO gets its own `default` persona —
before per-account persona ids only the first one did."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from chalybclip.db import Database, PersonasRepo
from chalybclip.integrations.chalyb.sso import sign_sso_token
from chalybclip.settings import get_settings
from chalybclip.tenancy import bound_tenant

_SECRET = "sso_default_persona_secret"


@pytest.fixture
def sso_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CHALYB_SSO_SECRET", _SECRET)
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


async def test_each_new_account_gets_default(
    client: httpx.AsyncClient, db: Database, sso_env: None
) -> None:
    for tid in ("ten_first_0001", "ten_second_0002", "ten_third_0003"):
        tok = sign_sso_token(user_id=f"u_{tid}", email=f"{tid}@x.co", tenant_id=tid, secret=_SECRET)
        r = await client.get(f"/auth/sso?token={tok}", follow_redirects=False)
        assert r.status_code == 303
        with bound_tenant(tid):
            p = await PersonasRepo(db).get("default")
        assert p is not None and p.tenant_id == tid
