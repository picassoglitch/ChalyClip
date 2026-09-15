"""REST: /webhooks/{id}/rotate-secret + /webhooks/{id}/secrets."""

from __future__ import annotations

import httpx

from .conftest import auth


async def test_rotate_secret_returns_new_value_once(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    create = await client.post(
        "/webhooks",
        json={"url": "https://example.com/x", "types": []},
        headers=auth(tenants["alice"]["token"]),
    )
    sub_id = create.json()["id"]
    original_secret = create.json()["secret"]

    rotate = await client.post(
        f"/webhooks/{sub_id}/rotate-secret",
        json={"grace_s": 3600},
        headers=auth(tenants["alice"]["token"]),
    )
    assert rotate.status_code == 200
    body = rotate.json()
    assert body["id"] == sub_id
    new_secret = body["secret"]
    assert new_secret != original_secret
    assert len(new_secret) == 64  # 32 hex bytes
    # Prior secret expiry surfaces so the caller knows when to update.
    assert body["prior_secret_expires_at"]


async def test_list_active_secrets_after_rotation(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    create = await client.post(
        "/webhooks",
        json={"url": "https://example.com/y", "types": []},
        headers=auth(tenants["alice"]["token"]),
    )
    sub_id = create.json()["id"]
    original_secret = create.json()["secret"]

    # Rotate twice with reasonable grace windows.
    for _ in range(2):
        await client.post(
            f"/webhooks/{sub_id}/rotate-secret",
            json={"grace_s": 3600},
            headers=auth(tenants["alice"]["token"]),
        )

    active = await client.get(
        f"/webhooks/{sub_id}/secrets", headers=auth(tenants["alice"]["token"])
    )
    assert active.status_code == 200
    rows = active.json()
    # Two rotations -> two prior secrets in the active window.
    assert len(rows) == 2
    # The original secret should appear among the rotated-out values.
    assert original_secret in {r["secret"] for r in rows}


async def test_rotate_unknown_subscription_returns_404(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    r = await client.post(
        "/webhooks/whk_does_not_exist/rotate-secret",
        json={},
        headers=auth(tenants["alice"]["token"]),
    )
    assert r.status_code == 404


async def test_list_secrets_for_other_tenants_subscription_returns_404(
    client: httpx.AsyncClient, tenants: dict[str, dict[str, str]]
) -> None:
    create = await client.post(
        "/webhooks",
        json={"url": "https://example.com/a", "types": []},
        headers=auth(tenants["alice"]["token"]),
    )
    sub_id = create.json()["id"]
    r = await client.get(
        f"/webhooks/{sub_id}/secrets", headers=auth(tenants["bob"]["token"])
    )
    assert r.status_code == 404


async def test_rotate_requires_full_scope(
    client: httpx.AsyncClient, db, tenants: dict[str, dict[str, str]]  # type: ignore[no-untyped-def]
) -> None:
    """A read-only token can't rotate."""
    from chalybclip.db import ApiTokensRepo
    from chalybclip.tenancy import bound_tenant, hash_token, mint_token

    create = await client.post(
        "/webhooks",
        json={"url": "https://example.com/c", "types": []},
        headers=auth(tenants["alice"]["token"]),
    )
    sub_id = create.json()["id"]

    # Mint a read-only token for Alice.
    with bound_tenant(tenants["alice"]["id"]):
        raw, _ = mint_token()
        await ApiTokensRepo(db).create(hash_=hash_token(raw), scope="read")

    r = await client.post(
        f"/webhooks/{sub_id}/rotate-secret",
        json={},
        headers=auth(raw),
    )
    assert r.status_code == 403
