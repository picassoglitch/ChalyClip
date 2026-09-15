"""WebhookSecretsRepo — rotate + list_active + purge_expired + tenancy."""

from __future__ import annotations

import datetime as _dt

import pytest

from chalybclip.db import (
    Database,
    TenantsRepo,
    WebhookSecretsRepo,
    WebhookSubscriptionsRepo,
)
from chalybclip.errors import ChalybClipError, TenancyError
from chalybclip.tenancy import bound_tenant


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


async def _seed_subscription(db: Database, tenant_id: str, *, secret: str = "old_secret") -> str:
    with bound_tenant(tenant_id):
        sub = await WebhookSubscriptionsRepo(db).create(
            url="https://example.com/x", types=[], secret=secret
        )
    return sub.id


async def test_rotate_writes_old_secret_to_versions_and_swaps_current(
    migrated_db: Database,
) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    sub_id = await _seed_subscription(migrated_db, tenant.id, secret="OLD")
    with bound_tenant(tenant.id):
        repo = WebhookSecretsRepo(migrated_db)
        out = await repo.rotate(sub_id, new_secret="NEW", ttl_s=3600)
    assert out == "NEW"

    # Subscription's current secret advanced.
    with bound_tenant(tenant.id):
        sub = await WebhookSubscriptionsRepo(migrated_db).get(sub_id)
    assert sub is not None
    assert sub.secret == "NEW"

    # Old secret persisted in the version table with a future expiry.
    with bound_tenant(tenant.id):
        active = await WebhookSecretsRepo(migrated_db).list_active_for_subscription(sub_id)
    assert len(active) == 1
    assert active[0].secret == "OLD"


async def test_rotate_unknown_subscription_raises(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    with bound_tenant(tenant.id), pytest.raises(ChalybClipError, match="not found"):
        await WebhookSecretsRepo(migrated_db).rotate(
            "whk_does_not_exist", new_secret="X", ttl_s=60
        )


async def test_list_active_drops_expired_versions(migrated_db: Database) -> None:
    """Expired secrets do NOT show in `list_active`."""
    tenant = await TenantsRepo(migrated_db).create(name="A")
    sub_id = await _seed_subscription(migrated_db, tenant.id, secret="OLD")
    with bound_tenant(tenant.id):
        # Rotate with a tiny ttl, then walk past it.
        await WebhookSecretsRepo(migrated_db).rotate(sub_id, new_secret="NEW", ttl_s=-1)
        active = await WebhookSecretsRepo(migrated_db).list_active_for_subscription(sub_id)
    # The version row exists but is already expired -> filtered out.
    assert active == []


async def test_purge_expired_deletes_past_grace_rows(migrated_db: Database) -> None:
    tenant = await TenantsRepo(migrated_db).create(name="A")
    sub_id = await _seed_subscription(migrated_db, tenant.id, secret="OLD")
    with bound_tenant(tenant.id):
        # Two rotations - the first instantly expires, the second stays alive.
        await WebhookSecretsRepo(migrated_db).rotate(sub_id, new_secret="MID", ttl_s=-1)
        await WebhookSecretsRepo(migrated_db).rotate(sub_id, new_secret="NEW", ttl_s=3600)
        purged = await WebhookSecretsRepo(migrated_db).purge_expired()
        active = await WebhookSecretsRepo(migrated_db).list_active_for_subscription(sub_id)
    assert purged == 1
    assert len(active) == 1
    assert active[0].secret == "MID"


async def test_rotate_requires_bound_tenant(migrated_db: Database) -> None:
    with pytest.raises(TenancyError, match="no tenant bound"):
        await WebhookSecretsRepo(migrated_db).rotate("whk_x", new_secret="N", ttl_s=60)


async def test_secrets_isolated_per_tenant(migrated_db: Database) -> None:
    """Bob can't rotate or read Alice's secrets."""
    alice = await TenantsRepo(migrated_db).create(name="Alice")
    bob = await TenantsRepo(migrated_db).create(name="Bob")
    sub_id = await _seed_subscription(migrated_db, alice.id, secret="ALICES")

    # Alice rotates in Alice's bound context.
    with bound_tenant(alice.id):
        await WebhookSecretsRepo(migrated_db).rotate(sub_id, new_secret="NEW", ttl_s=3600)

    # Bob can't see the rotation history.
    with bound_tenant(bob.id):
        active = await WebhookSecretsRepo(migrated_db).list_active_for_subscription(sub_id)
    assert active == []

    # Bob's rotate against Alice's subscription id raises.
    with bound_tenant(bob.id), pytest.raises(ChalybClipError, match="not found"):
        await WebhookSecretsRepo(migrated_db).rotate(
            sub_id, new_secret="HIJACK", ttl_s=60
        )

    # Alice's current secret unchanged by Bob's failed attempt.
    with bound_tenant(alice.id):
        sub = await WebhookSubscriptionsRepo(migrated_db).get(sub_id)
    assert sub is not None
    assert sub.secret == "NEW"
