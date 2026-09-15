"""Repository layer — one class per table, tenancy-enforced at the SQL boundary.

Every read and write of tenant-scoped data calls `current_tenant_id()` and
includes `tenant_id = ?` in the WHERE / VALUES clause. Defense in depth:
even if a handler forgot to bind the right tenant, the repo refuses to
return another tenant's rows.

Phase 0 service models (Stream, Candidate, Clip, ...) are converted
to/from these DB rows inside Task 1's persistence migration. For Task 0
the repos accept and return the DB models in `chalybclip.db.models`.

Auth path note: `ApiTokensRepo.lookup_by_hash` is the ONE tenant-unscoped
read in this module — it has to be, because we don't know the tenant
until the lookup succeeds. Every other read enforces tenancy.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import TYPE_CHECKING, Any, TypeVar

import aiosqlite
from pydantic import BaseModel

from chalybclip.errors import ChalybClipError, TenancyError
from chalybclip.ids import new_id
from chalybclip.tenancy import current_tenant_id

from .connection import Database

if TYPE_CHECKING:
    from chalybclip.publish.pacing import PlatformRule
from .models import (
    ApiTokenRow,
    BrandKitRow,
    CandidateRow,
    ChannelWatchRow,
    ClipRow,
    ConnectedAccount,
    CustomTriggerPhrases,
    DriveExportSettingsRow,
    DriveWatchRow,
    Event,
    HubPublishJobRow,
    LiveStreamKeyRow,
    LLMCallRow,
    PersonaRow,
    ProviderSpend,
    PublishJob,
    PublishMetric,
    SpeakerRow,
    StreamRow,
    StreamSpend,
    Tenant,
    TranscriptRow,
    User,
    VariantRow,
    VodSpeakerRow,
    WebhookSecretVersion,
    WebhookSubscription,
    ZernioEventRow,
    ZernioPublishRow,
)

_M = TypeVar("_M", bound=BaseModel)


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


class _Unset:
    """Sentinel singleton for kwargs where `None` is a legal "clear to NULL" value.

    Used by `TenantsRepo.set_budget` so callers can distinguish "leave the
    column alone" (omit kwarg) from "set the column to NULL" (pass None).
    """

    _instance: _Unset | None = None

    def __new__(cls) -> _Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "_UNSET"


_UNSET = _Unset()


class PlatformSettingsRepo:
    """Global key/value store for site-wide operator settings.

    Deliberately NOT tenant-scoped — like TenantsRepo, this is one of the
    few tables that holds platform-level rows rather than per-tenant data.
    Backs the admin "Site / Landing" page (operator-editable landing copy
    such as the displayed price) so those values change without a redeploy.

    Keys are free-form strings; values are stored as TEXT and the caller
    owns any parsing. Reads are safe to call from public routes (the
    landing page reads the price here at render time).
    """

    def __init__(self, db: Database):
        self._db = db

    async def get(self, key: str, default: str | None = None) -> str | None:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT value FROM platform_settings WHERE key = ?", (key,)
        )
        row = await cur.fetchone()
        return row[0] if row is not None else default

    async def get_many(self, keys: list[str]) -> dict[str, str]:
        """Fetch several keys in one round-trip. Missing keys are simply
        absent from the returned dict (callers use dict.get with a default)."""
        if not keys:
            return {}
        conn = await self._db.connect()
        placeholders = ",".join("?" for _ in keys)
        cur = await conn.execute(
            f"SELECT key, value FROM platform_settings WHERE key IN ({placeholders})",
            tuple(keys),
        )
        rows = await cur.fetchall()
        return {row[0]: row[1] for row in rows}

    async def set(self, key: str, value: str) -> None:
        """Upsert a single key. Portable ON CONFLICT upsert (works on both
        SQLite >=3.24 and Postgres) so it's create-or-update in one statement."""
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO platform_settings (key, value, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT (key) DO UPDATE SET "
            "value = excluded.value, updated_at = excluded.updated_at",
            (key, value, _now()),
        )
        await conn.commit()

    async def delete(self, key: str) -> None:
        conn = await self._db.connect()
        await conn.execute("DELETE FROM platform_settings WHERE key = ?", (key,))
        await conn.commit()


class TenantsRepo:
    """Tenants table — the one place where we don't filter by tenant_id."""

    def __init__(self, db: Database):
        self._db = db

    async def create(self, *, tenant_id: str | None = None, name: str) -> Tenant:
        tid = tenant_id or new_id("ten")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO tenants (id, name, created_at) VALUES (?, ?, ?)",
            (tid, name, _now()),
        )
        await conn.commit()
        return await self.get_or_raise(tid)

    async def get(self, tenant_id: str) -> Tenant | None:
        conn = await self._db.connect()
        # NOTE: every SELECT against `tenants` MUST list `external_user_id` —
        # the Tenant Pydantic model has it as an optional field (NX.1) and
        # callers (balance.py, reporter.py, service.py) read it directly.
        # Forgetting it here = silent AttributeError on every LLM call which
        # is what kept the token chip stuck on "— tokens · refresh" until
        # this fix landed.
        cur = await conn.execute(
            "SELECT id, name, created_at, daily_llm_budget_usd_micros, "
            "daily_publish_limit, rescore_concurrency_cap, "
            "retention_vod_days, retention_clip_days, retention_transcript_days, "
            "tier, status, external_user_id, "
            "cached_balance_remaining, cached_balance_unlimited, "
            "cached_balance_monthly_used, cached_balance_at, "
            "last_usage_report_at, last_usage_report_ok, last_usage_report_error, "
            "upload_post_profile_username, zernio_profile_id, zernio_profile_name "
            "FROM tenants WHERE id = ?",
            (tenant_id,),
        )
        return _model(Tenant, await cur.fetchone())

    async def find_by_external_user_id(self, external_user_id: str) -> Tenant | None:
        """Look up a tenant by its Chalyb cross-system user id.

        Returns None if no tenant has claimed that external id yet. Used by
        the /api/admin/tenants endpoint to make provisioning idempotent:
        re-calls from Chalyb (retries, admin re-grants) find the existing
        tenant instead of creating a duplicate.
        """
        if not external_user_id:
            return None
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, name, created_at, daily_llm_budget_usd_micros, "
            "daily_publish_limit, rescore_concurrency_cap, "
            "retention_vod_days, retention_clip_days, retention_transcript_days, "
            "tier, status, external_user_id, "
            "cached_balance_remaining, cached_balance_unlimited, "
            "cached_balance_monthly_used, cached_balance_at, "
            "last_usage_report_at, last_usage_report_ok, last_usage_report_error, "
            "upload_post_profile_username, zernio_profile_id, zernio_profile_name "
            "FROM tenants WHERE external_user_id = ?",
            (external_user_id,),
        )
        return _model(Tenant, await cur.fetchone())

    async def find_by_user_email(self, email: str) -> Tenant | None:
        """Look up a tenant by an email registered against ANY of its users.

        Used by the self-healing provision path: when Chalyb posts a
        provisioning call for an external_user_id we've never seen, we
        check whether a CLI-era tenant already exists for that email and
        claim it (backfilling external_user_id) rather than creating a
        duplicate tenant for the same human. Email is the only stable
        cross-system identifier we have for legacy rows.

        Multiple tenants COULD theoretically own the same email (the users
        table doesn't UNIQUE on it). If that happens we return the OLDEST
        tenant (lowest created_at) — that's the most likely target since
        it's the original CLI-created one before any later duplicates.
        """
        if not email:
            return None
        # Lower-case match — emails are case-insensitive per RFC 5321 §2.4.
        normalized = email.strip().lower()
        if not normalized:
            return None
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT t.id, t.name, t.created_at, t.daily_llm_budget_usd_micros, "
            "t.daily_publish_limit, t.rescore_concurrency_cap, "
            "t.retention_vod_days, t.retention_clip_days, t.retention_transcript_days, "
            "t.tier, t.status, t.external_user_id, "
            "t.cached_balance_remaining, t.cached_balance_unlimited, "
            "t.cached_balance_monthly_used, t.cached_balance_at, "
            "t.last_usage_report_at, t.last_usage_report_ok, t.last_usage_report_error, "
            "t.upload_post_profile_username, t.zernio_profile_id, t.zernio_profile_name "
            "FROM tenants t "
            "INNER JOIN users u ON u.tenant_id = t.id "
            "WHERE LOWER(u.email) = ? "
            "ORDER BY t.created_at ASC "
            "LIMIT 1",
            (normalized,),
        )
        return _model(Tenant, await cur.fetchone())

    async def set_status(self, tenant_id: str, status: str) -> None:
        """Flip tenant status. Used by /api/admin/tenants/{id}/status which
        Chalyb calls when a PRO subscriber swaps their live engine — the
        previously-active engine (this one) is paused so they can't cross-
        use multiple engines on a single-slot plan. Valid values are bound
        by the CHECK constraint in migration 015."""
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE tenants SET status = ? WHERE id = ?",
            (status, tenant_id),
        )
        await conn.commit()

    async def set_balance_cache(
        self,
        tenant_id: str,
        *,
        remaining: int,
        unlimited: bool,
        monthly_used: int,
        at_iso: str,
    ) -> None:
        """Stash the latest Chalyb balance numbers on this tenant row so
        templates can render the chip without a network call. Called by the
        outbound usage reporter after each successful POST to Chalyb."""
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE tenants SET "
            "cached_balance_remaining = ?, "
            "cached_balance_unlimited = ?, "
            "cached_balance_monthly_used = ?, "
            "cached_balance_at = ? "
            "WHERE id = ?",
            (remaining, 1 if unlimited else 0, monthly_used, at_iso, tenant_id),
        )
        await conn.commit()
        # Push the change to any open balance-chip SSE connections for this
        # tenant so the chip updates instantly (no client polling). Best-effort
        # and never raises — a failed notify must not fail the balance write.
        try:
            from chalybclip.events.balance_bus import balance_bus

            balance_bus.publish(
                tenant_id,
                {
                    "remaining": remaining,
                    "unlimited": bool(unlimited),
                    "monthly_used": monthly_used,
                    "at": at_iso,
                },
            )
        except Exception:  # pragma: no cover — notify is best-effort
            pass

    async def set_usage_report_status(
        self,
        tenant_id: str,
        *,
        ok: bool,
        at_iso: str,
        error: str | None = None,
    ) -> None:
        """Token T1 — record the outcome of the most recent outbound
        usage report. Called by the reporter on every terminal path
        (success or real failure) so the chip / diag can surface a
        failing balance-sync instead of a silently stale number.

        `error` is a short reason on failure; None on success.
        """
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE tenants SET "
            "last_usage_report_at = ?, "
            "last_usage_report_ok = ?, "
            "last_usage_report_error = ? "
            "WHERE id = ?",
            (at_iso, 1 if ok else 0, (error if not ok else None), tenant_id),
        )
        await conn.commit()

    async def set_external_user_id(self, tenant_id: str, external_user_id: str) -> None:
        """Bind an existing tenant to a Chalyb user. One-shot."""
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE tenants SET external_user_id = ? WHERE id = ?",
            (external_user_id, tenant_id),
        )
        await conn.commit()

    async def set_upload_post_profile_username(
        self, tenant_id: str, username: str | None,
    ) -> None:
        """Persist (or clear) the upload-post 'user profile' username
        for a tenant.

        Set once on first connect/publish via ensure_profile_for_tenant(),
        explicitly via the /claim endpoint, or cleared via the /unlink
        endpoint. Subsequent upload-post API calls reuse this value as
        the `user` field on every request. Pass None or empty string
        to clear (forget the binding without touching upload-post's
        side).
        """
        # Normalize empty string to NULL — the model treats both as
        # "unbound" but NULL is the canonical storage form.
        value = username if username else None
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE tenants SET upload_post_profile_username = ? WHERE id = ?",
            (value, tenant_id),
        )
        await conn.commit()

    async def set_zernio_profile(
        self,
        tenant_id: str,
        *,
        profile_id: str | None,
        profile_name: str | None = None,
    ) -> None:
        """Persist (or clear) the Zernio profile binding for a tenant.

        Set when the operator creates a profile (create_profile_for_tenant)
        or binds an existing one (/accounts/claim); cleared via /unlink.
        Zernio scopes connect + accounts by `profile_id`; `profile_name`
        is the display name. Pass None/empty profile_id to clear BOTH
        columns (forget the binding without touching Zernio's side).
        """
        # Normalize empty string to NULL — the model treats both as
        # "unbound" but NULL is the canonical storage form. Clearing the
        # id always clears the name too (they're one binding).
        pid = profile_id or None
        pname = (profile_name or None) if pid else None
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE tenants SET zernio_profile_id = ?, zernio_profile_name = ? "
            "WHERE id = ?",
            (pid, pname, tenant_id),
        )
        await conn.commit()

    async def get_or_raise(self, tenant_id: str) -> Tenant:
        t = await self.get(tenant_id)
        if t is None:
            raise ChalybClipError(f"tenant not found: {tenant_id}")
        return t

    async def find_by_zernio_profile(self, profile_id: str) -> Tenant | None:
        """Tenant-FREE reverse lookup: Zernio profileId → tenant.

        Inbound Zernio webhooks carry a profileId but no tenant (the
        signature is the transport auth); this is the resolution step.
        One profile maps to one tenant in our model — the claim flow
        rejects binding a profileId to a second tenant."""
        if not profile_id:
            return None
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id FROM tenants WHERE zernio_profile_id = ?",
            (profile_id,),
        )
        row = await cur.fetchone()
        return await self.get(row["id"]) if row else None

    async def list_all(self) -> list[Tenant]:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, name, created_at, daily_llm_budget_usd_micros, "
            "daily_publish_limit, rescore_concurrency_cap, "
            "retention_vod_days, retention_clip_days, retention_transcript_days, "
            "tier, status, external_user_id, "
            "cached_balance_remaining, cached_balance_unlimited, "
            "cached_balance_monthly_used, cached_balance_at, "
            "last_usage_report_at, last_usage_report_ok, last_usage_report_error, "
            "upload_post_profile_username, zernio_profile_id, zernio_profile_name "
            "FROM tenants ORDER BY created_at"
        )
        return [Tenant.model_validate(dict(r)) for r in await cur.fetchall()]

    async def set_retention(
        self,
        tenant_id: str,
        *,
        retention_vod_days: int | None | _Unset = _UNSET,
        retention_clip_days: int | None | _Unset = _UNSET,
        retention_transcript_days: int | None | _Unset = _UNSET,
    ) -> Tenant:
        """Update retention windows for one tenant.

        Same sentinel pattern as `set_budget`: a passed `None` clears the
        column to NULL (i.e. the tenant inherits the system default).
        Omitting a kwarg leaves the existing value untouched.
        """
        sets: list[str] = []
        values: list[object] = []
        if not isinstance(retention_vod_days, _Unset):
            sets.append("retention_vod_days = ?")
            values.append(retention_vod_days)
        if not isinstance(retention_clip_days, _Unset):
            sets.append("retention_clip_days = ?")
            values.append(retention_clip_days)
        if not isinstance(retention_transcript_days, _Unset):
            sets.append("retention_transcript_days = ?")
            values.append(retention_transcript_days)
        if not sets:
            return await self.get_or_raise(tenant_id)
        values.append(tenant_id)
        conn = await self._db.connect()
        await conn.execute(
            f"UPDATE tenants SET {', '.join(sets)} WHERE id = ?",
            tuple(values),
        )
        await conn.commit()
        return await self.get_or_raise(tenant_id)

    async def set_budget(
        self,
        tenant_id: str,
        *,
        daily_llm_budget_usd_micros: int | None | _Unset = _UNSET,
        daily_publish_limit: int | None | _Unset = _UNSET,
        rescore_concurrency_cap: int | _Unset = _UNSET,
    ) -> Tenant:
        """Update governor knobs.

        A passed `None` *clears* the cap to NULL (unlimited). To leave a
        column untouched, omit the kwarg (sentinel `_UNSET`).
        """
        sets: list[str] = []
        values: list[object] = []
        if not isinstance(daily_llm_budget_usd_micros, _Unset):
            sets.append("daily_llm_budget_usd_micros = ?")
            values.append(daily_llm_budget_usd_micros)
        if not isinstance(daily_publish_limit, _Unset):
            sets.append("daily_publish_limit = ?")
            values.append(daily_publish_limit)
        if not isinstance(rescore_concurrency_cap, _Unset):
            sets.append("rescore_concurrency_cap = ?")
            values.append(rescore_concurrency_cap)
        if not sets:
            return await self.get_or_raise(tenant_id)
        values.append(tenant_id)
        conn = await self._db.connect()
        await conn.execute(
            f"UPDATE tenants SET {', '.join(sets)} WHERE id = ?",
            tuple(values),
        )
        await conn.commit()
        return await self.get_or_raise(tenant_id)


class UsersRepo:
    """Users live under a tenant; reads filter on `current_tenant_id()`."""

    def __init__(self, db: Database):
        self._db = db

    async def create(self, *, email: str, role: str = "owner") -> User:
        tenant_id = current_tenant_id()
        user_id = new_id("usr")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO users (id, tenant_id, email, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, tenant_id, email, role, _now()),
        )
        await conn.commit()
        u = await self.get(user_id)
        assert u is not None  # just inserted
        return u

    async def get(self, user_id: str) -> User | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, tenant_id, email, role, created_at FROM users "
            "WHERE id = ? AND tenant_id = ?",
            (user_id, tenant_id),
        )
        return _model(User, await cur.fetchone())

    async def list_for_tenant(self) -> list[User]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, tenant_id, email, role, created_at FROM users "
            "WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        )
        return [User.model_validate(dict(r)) for r in await cur.fetchall()]


class ApiTokensRepo:
    """API tokens. Stores only the sha256 hash, never the raw token."""

    def __init__(self, db: Database):
        self._db = db

    async def create(self, *, hash_: str, scope: str = "full") -> ApiTokenRow:
        tenant_id = current_tenant_id()
        token_id = new_id("tok")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO api_tokens (id, tenant_id, hash, scope, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (token_id, tenant_id, hash_, scope, _now()),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT id, tenant_id, hash, scope, created_at, last_used_at "
            "FROM api_tokens WHERE id = ? AND tenant_id = ?",
            (token_id, tenant_id),
        )
        row = await cur.fetchone()
        assert row is not None
        return ApiTokenRow.model_validate(dict(row))

    async def lookup_by_hash(self, hash_: str) -> ApiTokenRow | None:
        """Auth path: tenant-unscoped lookup. Returns the row including tenant_id.

        This is the ONLY read in the repo layer that doesn't filter by
        tenant — by definition we don't know the tenant until this lookup
        succeeds. After it returns, the caller is expected to bind that
        tenant via `bound_tenant(...)` before issuing any further query.
        """
        if not hash_:
            raise TenancyError("empty token hash")
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, tenant_id, hash, scope, created_at, last_used_at "
            "FROM api_tokens WHERE hash = ?",
            (hash_,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        # Touch last_used_at so we can audit token usage.
        await conn.execute(
            "UPDATE api_tokens SET last_used_at = ? WHERE id = ?",
            (_now(), row["id"]),
        )
        await conn.commit()
        return ApiTokenRow.model_validate(dict(row))

    async def list_for_tenant(self) -> list[ApiTokenRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, tenant_id, hash, scope, created_at, last_used_at "
            "FROM api_tokens WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        )
        return [ApiTokenRow.model_validate(dict(r)) for r in await cur.fetchall()]


class PersonasRepo:
    def __init__(self, db: Database):
        self._db = db

    async def create(
        self,
        *,
        persona_id: str,
        name: str,
        primary_language: str,
        target_languages: list[str],
        voice_prompt: str,
        routing_tags: list[str] | None = None,
    ) -> PersonaRow:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO personas (id, tenant_id, name, primary_language, "
            "target_languages_json, voice_prompt, routing_tags_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                persona_id,
                tenant_id,
                name,
                primary_language,
                json.dumps(target_languages),
                voice_prompt,
                json.dumps(routing_tags or []),
                _now(),
            ),
        )
        await conn.commit()
        p = await self.get(persona_id)
        assert p is not None
        return p

    async def upsert(
        self,
        *,
        persona_id: str,
        name: str,
        primary_language: str,
        target_languages: list[str],
        voice_prompt: str,
        routing_tags: list[str] | None = None,
    ) -> PersonaRow:
        """Insert or replace. Used by the YAML→DB transition in Task 1."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO personas (id, tenant_id, name, primary_language, "
            "target_languages_json, voice_prompt, routing_tags_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "name = excluded.name, "
            "primary_language = excluded.primary_language, "
            "target_languages_json = excluded.target_languages_json, "
            "voice_prompt = excluded.voice_prompt, "
            "routing_tags_json = excluded.routing_tags_json",
            (
                persona_id,
                tenant_id,
                name,
                primary_language,
                json.dumps(target_languages),
                voice_prompt,
                json.dumps(routing_tags or []),
                _now(),
            ),
        )
        await conn.commit()
        p = await self.get(persona_id)
        assert p is not None
        return p

    async def get(self, persona_id: str) -> PersonaRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM personas WHERE id = ? AND tenant_id = ?",
            (persona_id, tenant_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return _persona_from_row(row)

    async def list_for_tenant(self) -> list[PersonaRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM personas WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        )
        return [_persona_from_row(r) for r in await cur.fetchall()]


def _persona_from_row(row: aiosqlite.Row) -> PersonaRow:
    d = dict(row)
    d["target_languages"] = json.loads(d.pop("target_languages_json"))
    d["routing_tags"] = json.loads(d.pop("routing_tags_json"))
    return PersonaRow.model_validate(d)


class StreamsRepo:
    def __init__(self, db: Database):
        self._db = db

    async def upsert(self, stream: StreamRow) -> StreamRow:
        """Insert if new; otherwise leave the row untouched (idempotent resume)."""
        if stream.tenant_id != current_tenant_id():
            raise TenancyError(
                f"stream tenant {stream.tenant_id!r} != bound {current_tenant_id()!r}"
            )
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO streams "
            "(id, tenant_id, vod_url, platform, title, channel, duration_s, "
            "source_video_path, source_audio_path, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            (
                stream.id,
                stream.tenant_id,
                stream.vod_url,
                stream.platform,
                stream.title,
                stream.channel,
                stream.duration_s,
                stream.source_video_path,
                stream.source_audio_path,
                stream.status,
                stream.created_at,
            ),
        )
        await conn.commit()
        existing = await self.get(stream.id)
        assert existing is not None
        return existing

    async def set_duration(self, stream_id: str, duration_s: float) -> None:
        """Backfill the real media duration onto a row created without one.

        `upsert` is INSERT OR IGNORE, so it never updates a row inserted
        earlier — e.g. a live stream the ChalybOBS webhook created with
        duration 0 (the real length isn't known until the recording is
        processed). The pipeline calls this after ffprobe so the streams
        list shows the actual length instead of 0s. Guarded on
        `duration_s <= 0` so a real value (e.g. from the live /ended
        webhook) is never clobbered by a slightly different ffprobe read."""
        if duration_s <= 0:
            return
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE streams SET duration_s = ? "
            "WHERE id = ? AND tenant_id = ? AND duration_s <= 0",
            (float(duration_s), stream_id, tenant_id),
        )
        await conn.commit()

    async def reconcile_metadata(
        self,
        stream_id: str,
        *,
        title: str | None,
        channel: str | None,
        duration_s: float,
    ) -> None:
        """Backfill real metadata onto a placeholder row after ingest.

        `upsert` is INSERT-OR-IGNORE, so a row pre-inserted as a placeholder
        (the URL-job flow at `dashboard.url_job_create` inserts
        status='pending', duration 0, no title) is NEVER updated by the
        pipeline's later upsert — it would display 'pending / 0s /
        url-as-title' forever even after a fully successful run. This fills
        the gaps WITHOUT clobbering values already set: title/channel only
        when NULL, duration only when <= 0 (same guard as `set_duration`),
        and status 'pending' -> 'ingested' (live / processing / done states
        are left untouched). Safe to call on the non-placeholder paths too —
        there it's a no-op since nothing is empty."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE streams SET "
            "title = COALESCE(title, ?), "
            "channel = COALESCE(channel, ?), "
            "duration_s = CASE WHEN duration_s <= 0 THEN ? ELSE duration_s END, "
            "status = CASE WHEN status = 'pending' THEN 'ingested' ELSE status END "
            "WHERE id = ? AND tenant_id = ?",
            (title, channel, float(duration_s or 0.0), stream_id, tenant_id),
        )
        await conn.commit()

    async def set_status(
        self, stream_id: str, *, status: str, only_if: str | None = None
    ) -> bool:
        """Flip a stream's status (tenant-scoped). When `only_if` is given the
        update is guarded — it changes the row ONLY if it is currently in that
        status, so a placeholder isn't clobbered if something else already
        advanced it. Returns True iff a row changed.

        Needed because `upsert` is INSERT-OR-IGNORE and never updates an
        existing row: e.g. flipping a detected-but-failed placeholder from
        'pending' to 'failed' can't go through upsert."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        if only_if is None:
            cur = await conn.execute(
                "UPDATE streams SET status = ? WHERE id = ? AND tenant_id = ?",
                (status, stream_id, tenant_id),
            )
        else:
            cur = await conn.execute(
                "UPDATE streams SET status = ? "
                "WHERE id = ? AND tenant_id = ? AND status = ?",
                (status, stream_id, tenant_id, only_if),
            )
        await conn.commit()
        return (cur.rowcount or 0) == 1

    async def get(self, stream_id: str) -> StreamRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM streams WHERE id = ? AND tenant_id = ?",
            (stream_id, tenant_id),
        )
        return _model(StreamRow, await cur.fetchone())

    async def find_by_vod_url(self, vod_url: str) -> StreamRow | None:
        """Return the OLDEST stream for (current tenant, vod_url), or None.

        Lets the channel poller reuse an already-ingested VOD's stream id
        instead of minting a new one and re-downloading — the dedup that
        keeps a re-poll (lost seen-set, or a prior pipeline failure the
        poller retried) from spawning duplicate stream rows. Oldest-first
        so repeated calls converge on one canonical stream even when
        historical duplicates already exist. Backed by idx_streams_tenant_vod
        (migration 047)."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM streams WHERE tenant_id = ? AND vod_url = ? "
            "ORDER BY created_at ASC LIMIT 1",
            (tenant_id, vod_url),
        )
        return _model(StreamRow, await cur.fetchone())

    async def list_for_tenant(self, *, limit: int = 10_000) -> list[StreamRow]:
        """List a tenant's streams, hard-capped to `limit` rows (default 10k).

        Bound for the same reason as CandidatesRepo / ClipsRepo .list_for_stream:
        a tenant with many years of streams would otherwise pull the whole
        history into memory on every dashboard render.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM streams WHERE tenant_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (tenant_id, int(limit)),
        )
        return [StreamRow.model_validate(dict(r)) for r in await cur.fetchall()]

    async def delete(self, stream_id: str) -> bool:
        """Slice O.11 — hard-delete a stream + everything that hangs off it.

        Returns True iff a row was deleted (the stream existed AND was
        owned by the current tenant). Foreign-keys ON DELETE CASCADE
        handle the rest: transcripts, candidates, clips (and through
        clips → variants, publish_jobs, publish_metrics, events) all
        get reaped automatically.

        Filesystem cleanup is the caller's responsibility — this method
        only touches the DB. The publish-jobs cascade means in-flight
        jobs lose their row too, so block deletion of a `running` stream
        at the route layer.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "DELETE FROM streams WHERE id = ? AND tenant_id = ?",
            (stream_id, tenant_id),
        )
        await conn.commit()
        return bool(cur.rowcount)


class LLMCallsRepo:
    """Append-only log of every LLM call for billing + audit."""

    def __init__(self, db: Database):
        self._db = db

    async def record(self, row: LLMCallRow) -> None:
        if row.tenant_id != current_tenant_id():
            raise TenancyError(
                f"llm_call tenant {row.tenant_id!r} != bound {current_tenant_id()!r}"
            )
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO llm_calls (id, tenant_id, purpose, provider, model, quality, "
            "input_tokens, output_tokens, cost_usd_micros, status, error, attempts, ts, "
            "stream_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.id,
                row.tenant_id,
                row.purpose,
                row.provider,
                row.model,
                row.quality,
                row.input_tokens,
                row.output_tokens,
                row.cost_usd_micros,
                row.status,
                row.error,
                row.attempts,
                row.ts,
                row.stream_id,
            ),
        )
        await conn.commit()

    async def cost_for_stream(self, stream_id: str) -> StreamSpend:
        """Token T3 — actual spend for ONE stream's run: total tokens +
        total USD cost, plus a per-provider breakdown. Drives the
        'what did this video actually cost' panel (vs the estimate)."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT provider, "
            "COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens, "
            "COALESCE(SUM(cost_usd_micros), 0) AS cost_um, "
            "COUNT(*) AS calls "
            "FROM llm_calls "
            "WHERE tenant_id = ? AND stream_id = ? AND status = 'ok' "
            "GROUP BY provider",
            (tenant_id, stream_id),
        )
        by_provider: list[ProviderSpend] = []
        total_tokens = 0
        total_cost = 0
        total_calls = 0
        for r in await cur.fetchall():
            d = dict(r)
            by_provider.append(
                ProviderSpend(
                    provider=d["provider"],
                    tokens=int(d["tokens"]),
                    cost_usd_micros=int(d["cost_um"]),
                    calls=int(d["calls"]),
                )
            )
            total_tokens += int(d["tokens"])
            total_cost += int(d["cost_um"])
            total_calls += int(d["calls"])
        return StreamSpend(
            stream_id=stream_id,
            total_tokens=total_tokens,
            total_cost_usd_micros=total_cost,
            total_calls=total_calls,
            by_provider=by_provider,
        )

    async def list_for_tenant(self, *, limit: int = 100) -> list[LLMCallRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM llm_calls WHERE tenant_id = ? ORDER BY ts DESC LIMIT ?",
            (tenant_id, limit),
        )
        return [LLMCallRow.model_validate(dict(r)) for r in await cur.fetchall()]

    async def total_spend_today_micros(self) -> int:
        """Sum of `cost_usd_micros` for the bound tenant since 00:00 UTC.

        The budget governor (P2 Task 1) calls this before each LLM request
        to decide whether the projected next-call cost would push today
        past `tenants.daily_llm_budget_usd_micros`.
        """
        tenant_id = current_tenant_id()
        cutoff = _start_of_utc_today()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT COALESCE(SUM(cost_usd_micros), 0) FROM llm_calls "
            "WHERE tenant_id = ? AND ts >= ?",
            (tenant_id, cutoff),
        )
        row = await cur.fetchone()
        return int(row[0]) if row else 0


async def _resolve_tenant_for_stream(
    db: Database, *, stream_id: str
) -> str | None:
    """Read the tenant_id of a stream row by id, or None if no such row.

    Used by the MediaMTX-webhook helpers below so each UPDATE can be
    tenant-scoped (CLAUDE.md rule #1) even though the webhook has no
    bound tenant of its own. If the lookup returns None the helper
    refuses the mutation rather than running an unbounded UPDATE.
    """
    conn = await db.connect()
    cur = await conn.execute(
        "SELECT tenant_id FROM streams WHERE id = ?", (stream_id,)
    )
    row = await cur.fetchone()
    return str(row["tenant_id"]) if row else None


async def _streams_repo_mark_live_started(
    db: Database, *, stream_id: str
) -> StreamRow | None:
    """Phase L.1 — flip a stream into the 'live' status.

    Free function (not on StreamsRepo) because the MediaMTX webhook
    runs without a bound tenant — it identifies the tenant via the
    stream key, then needs to update by id. We expose this here so
    the webhook handler can call it directly without going through
    the full tenant-binding ceremony.

    Tenant scope is derived from the row itself (lookup-then-update);
    the UPDATE includes WHERE tenant_id = ? so an attacker with the
    CHALYBCLIP_INTERNAL_SIGNING_SECRET cannot mutate a stream they
    cannot also enumerate (defense in depth — the row lookup is the
    real ownership gate, the WHERE clause is the safety net).

    Idempotent: calling twice for the same stream_id is a no-op past
    the first call (since the row already has is_live=1).
    """
    tenant_id = await _resolve_tenant_for_stream(db, stream_id=stream_id)
    if tenant_id is None:
        return None
    now = _now()
    conn = await db.connect()
    await conn.execute(
        "UPDATE streams "
        "SET is_live = 1, "
        "    live_started_at = COALESCE(live_started_at, ?), "
        "    status = 'live' "
        "WHERE id = ? AND tenant_id = ?",
        (now, stream_id, tenant_id),
    )
    await conn.commit()
    cur = await conn.execute(
        "SELECT * FROM streams WHERE id = ? AND tenant_id = ?",
        (stream_id, tenant_id),
    )
    row = await cur.fetchone()
    return StreamRow.model_validate(dict(row)) if row else None


async def _streams_repo_mark_live_ended(
    db: Database, *, stream_id: str, duration_s: float | None = None
) -> StreamRow | None:
    """Phase L.1 — flip a stream from 'live' to 'live_ended'.

    Same tenant-derived-from-row contract as `_streams_repo_mark_live_started`.
    Optionally accepts the final duration from MediaMTX's
    runOnNotReady webhook payload so the streams row stops showing 0.
    """
    tenant_id = await _resolve_tenant_for_stream(db, stream_id=stream_id)
    if tenant_id is None:
        return None
    now = _now()
    conn = await db.connect()
    if duration_s is not None:
        await conn.execute(
            "UPDATE streams "
            "SET is_live = 0, "
            "    live_ended_at = ?, "
            "    status = 'live_ended', "
            "    duration_s = ? "
            "WHERE id = ? AND tenant_id = ?",
            (now, float(duration_s), stream_id, tenant_id),
        )
    else:
        await conn.execute(
            "UPDATE streams "
            "SET is_live = 0, "
            "    live_ended_at = ?, "
            "    status = 'live_ended' "
            "WHERE id = ? AND tenant_id = ?",
            (now, stream_id, tenant_id),
        )
    await conn.commit()
    cur = await conn.execute(
        "SELECT * FROM streams WHERE id = ? AND tenant_id = ?",
        (stream_id, tenant_id),
    )
    row = await cur.fetchone()
    return StreamRow.model_validate(dict(row)) if row else None


async def _streams_repo_try_claim_for_processing(
    db: Database, *, stream_id: str, from_status: str = "live_ended"
) -> bool:
    """Phase L.2 — atomically claim a just-ended live stream for the
    auto-clip pipeline.

    Flips status `from_status` -> 'processing' ONLY if the row is still in
    `from_status`. Returns True iff THIS call won the claim. This is the
    idempotency guard for the auto-clip kickoff: MediaMTX may deliver the
    'ended' webhook more than once (retries / flaky TCP), and we must not
    launch the (paid) pipeline twice. The first webhook claims and
    schedules; every later one gets False and skips.

    Tenant scope derived from the row itself (lookup-then-update),
    matching the contract of the mark_live_* helpers above.
    """
    tenant_id = await _resolve_tenant_for_stream(db, stream_id=stream_id)
    if tenant_id is None:
        return False
    conn = await db.connect()
    cur = await conn.execute(
        "UPDATE streams SET status = 'processing' "
        "WHERE id = ? AND tenant_id = ? AND status = ?",
        (stream_id, tenant_id, from_status),
    )
    await conn.commit()
    return (cur.rowcount or 0) == 1


async def _streams_repo_set_status(
    db: Database, *, stream_id: str, status: str
) -> None:
    """Phase L.2 — tenant-free stream status update (same invocation
    contract as the mark_live_* / claim helpers). The live runner calls
    this to flip a stream to a terminal status ('done') once its clip
    pipeline finishes — otherwise it stays at the 'processing' the autoclip
    claim set, and the dashboard shows 'Analyzing…' forever with clips on
    disk."""
    conn = await db.connect()
    await conn.execute(
        "UPDATE streams SET status = ? WHERE id = ?",
        (status, stream_id),
    )
    await conn.commit()


class LiveStreamKeysRepo:
    """Phase L.1 — per-tenant RTMP stream keys.

    Operator generates one via the live dashboard, copies the RTMP URL
    + key into OBS. MediaMTX validates the key on every push via the
    /api/internal/live/authorize webhook (calls `find_by_value` here).

    Rotation: `rotate_for_tenant` is the one-shot "generate a new key,
    revoke any existing one" path the dashboard's rotate button uses.
    Never mutates `key_value` on existing rows — old key stays in the
    table marked revoked for audit, new key is a new row.
    """

    def __init__(self, db: Database):
        self._db = db

    @staticmethod
    def _new_key_value() -> str:
        """Generate a fresh RTMP stream key. Slug shape `slk_<token>`
        so the key is grep-able in logs (the secret-looking `_token`
        suffix is the unguessable bit; the `slk_` prefix is for
        humans). 32 url-safe bytes = ~43 char token; full key length
        is ~47 chars, fits comfortably in OBS's URL field."""
        import secrets
        return f"slk_{secrets.token_urlsafe(32)}"

    async def get_active_for_tenant(self) -> LiveStreamKeyRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM live_stream_keys "
            "WHERE tenant_id = ? AND revoked_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1",
            (tenant_id,),
        )
        row = await cur.fetchone()
        return LiveStreamKeyRow.model_validate(dict(row)) if row else None

    async def find_by_value(self, key_value: str) -> LiveStreamKeyRow | None:
        """Used by the MediaMTX auth webhook. NO tenant binding —
        this is the entry point that DECIDES tenant identity from
        the key value, so binding would create a chicken-and-egg.
        Caller (the webhook handler) is responsible for not echoing
        the looked-up tenant outside the live ingest scope.
        """
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM live_stream_keys WHERE key_value = ?",
            (key_value,),
        )
        row = await cur.fetchone()
        return LiveStreamKeyRow.model_validate(dict(row)) if row else None

    async def rotate_for_tenant(self) -> LiveStreamKeyRow:
        """Generate a fresh key for the bound tenant; revoke any
        existing active key in the same transaction. Returns the new
        row.

        Idempotent in the sense that calling twice still leaves one
        active key; the second call just rotates AGAIN. If the
        operator clicks rotate by accident their OBS streams need to
        update — surface that in the UI copy."""
        tenant_id = current_tenant_id()
        now = _now()
        new_id = new_id_with_prefix("lsk")
        new_key = self._new_key_value()
        conn = await self._db.connect()
        # Revoke any existing active key for this tenant.
        await conn.execute(
            "UPDATE live_stream_keys SET revoked_at = ? "
            "WHERE tenant_id = ? AND revoked_at IS NULL",
            (now, tenant_id),
        )
        # Insert the new one.
        await conn.execute(
            "INSERT INTO live_stream_keys "
            "(id, tenant_id, key_value, created_at) VALUES (?, ?, ?, ?)",
            (new_id, tenant_id, new_key, now),
        )
        await conn.commit()
        out = await self.get_active_for_tenant()
        assert out is not None  # we just inserted it
        return out

    async def touch_last_used(self, key_id: str) -> None:
        """Update last_used_at on a successful authorize webhook.
        Best-effort: a write failure here doesn't block the push."""
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE live_stream_keys SET last_used_at = ? WHERE id = ?",
            (_now(), key_id),
        )
        await conn.commit()


def new_id_with_prefix(prefix: str) -> str:
    """ULID with an alpha prefix, e.g. `lsk_01XXX`. Mirrors `new_id`
    but lets callers pass arbitrary prefixes for new entity types
    that haven't been blessed into the central new_id helper yet."""
    return new_id(prefix)


class EventsRepo:
    """Append-only event log. Every state transition lands here."""

    def __init__(self, db: Database):
        self._db = db

    async def emit(
        self,
        *,
        type: str,  # matches the column name
        payload: dict[str, Any] | None = None,
    ) -> Event:
        tenant_id = current_tenant_id()
        event_id = new_id("evt")
        ts = _now()
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO events (id, tenant_id, type, payload_json, ts) VALUES (?, ?, ?, ?, ?)",
            (event_id, tenant_id, type, json.dumps(payload or {}), ts),
        )
        await conn.commit()
        return Event(
            id=event_id,
            tenant_id=tenant_id,
            type=type,
            payload=payload or {},
            ts=ts,
        )

    async def list_for_tenant(
        self,
        *,
        type: str | None = None,
        limit: int = 100,
    ) -> list[Event]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        if type is None:
            cur = await conn.execute(
                "SELECT id, tenant_id, type, payload_json, ts FROM events "
                "WHERE tenant_id = ? ORDER BY ts DESC LIMIT ?",
                (tenant_id, limit),
            )
        else:
            cur = await conn.execute(
                "SELECT id, tenant_id, type, payload_json, ts FROM events "
                "WHERE tenant_id = ? AND type = ? ORDER BY ts DESC LIMIT ?",
                (tenant_id, type, limit),
            )
        return [_event_from_row(r) for r in await cur.fetchall()]


def _event_from_row(row: aiosqlite.Row) -> Event:
    d = dict(row)
    d["payload"] = json.loads(d.pop("payload_json") or "{}")
    return Event.model_validate(d)


def _model(cls: type[_M], row: aiosqlite.Row | None) -> _M | None:
    """Round-trip an aiosqlite Row to a Pydantic model, or return None."""
    if row is None:
        return None
    return cls.model_validate(dict(row))


# ---------------------------------------------------------------------------
# Pipeline repos: transcripts, candidates, clips, variants.
#
# All four follow the same pattern as StreamsRepo: every read filters by
# `current_tenant_id()`; every write asserts the row's tenant_id matches.
# Bulk methods are idempotent on insert conflicts so resuming a partially-
# committed pipeline doesn't duplicate rows.
# ---------------------------------------------------------------------------


class TranscriptsRepo:
    """One transcript per stream (PK is stream_id)."""

    def __init__(self, db: Database):
        self._db = db

    async def upsert(self, row: TranscriptRow) -> TranscriptRow:
        if row.tenant_id != current_tenant_id():
            raise TenancyError(
                f"transcript tenant {row.tenant_id!r} != bound {current_tenant_id()!r}"
            )
        conn = await self._db.connect()
        # Replace any existing transcript for this stream — Whisper output
        # for the same stream is deterministic enough that a re-run is fine.
        await conn.execute(
            "INSERT INTO transcripts "
            "(stream_id, tenant_id, language, duration_s, model, segments_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(stream_id) DO UPDATE SET "
            "language = excluded.language, "
            "duration_s = excluded.duration_s, "
            "model = excluded.model, "
            "segments_json = excluded.segments_json",
            (
                row.stream_id,
                row.tenant_id,
                row.language,
                row.duration_s,
                row.model,
                row.segments_json,
                row.created_at,
            ),
        )
        await conn.commit()
        existing = await self.get(row.stream_id)
        assert existing is not None
        return existing

    async def get(self, stream_id: str) -> TranscriptRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM transcripts WHERE stream_id = ? AND tenant_id = ?",
            (stream_id, tenant_id),
        )
        return _model(TranscriptRow, await cur.fetchone())


class CandidatesRepo:
    """Detected candidates. Insert is idempotent on (id) so resumes work."""

    def __init__(self, db: Database):
        self._db = db

    async def upsert_many(self, rows: list[CandidateRow]) -> int:
        """Insert candidates, ignoring conflicts. Returns count inserted (or 0)."""
        if not rows:
            return 0
        bound = current_tenant_id()
        for row in rows:
            if row.tenant_id != bound:
                raise TenancyError(f"candidate tenant {row.tenant_id!r} != bound {bound!r}")
        conn = await self._db.connect()
        await conn.executemany(
            "INSERT INTO candidates "
            "(id, stream_id, tenant_id, ts, score, reason, evidence_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [
                (
                    r.id,
                    r.stream_id,
                    r.tenant_id,
                    r.ts,
                    r.score,
                    r.reason,
                    json.dumps(r.evidence),
                    r.created_at,
                )
                for r in rows
            ],
        )
        await conn.commit()
        return len(rows)

    async def list_for_stream(
        self, stream_id: str, *, limit: int = 10_000
    ) -> list[CandidateRow]:
        """List candidates for a stream, hard-capped to `limit` rows.

        Default of 10 000 protects against OOM when a long stream
        produces a runaway candidate set (~5 000 is already a busy
        90-minute show). Callers that need higher can opt in
        explicitly; nothing in the dashboard currently does.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM candidates WHERE tenant_id = ? AND stream_id = ? "
            "ORDER BY ts LIMIT ?",
            (tenant_id, stream_id, int(limit)),
        )
        return [_candidate_from_row(r) for r in await cur.fetchall()]

    async def update_rescore(
        self,
        candidate_id: str,
        *,
        rescore_score: float | None,
        rescore_reason: str | None,
        rescore_model: str | None,
    ) -> CandidateRow:
        """Persist a vision-LLM rescore verdict (P2 Task 3 writes here).

        Pass None for all three to clear a prior verdict.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE candidates SET rescore_score = ?, rescore_reason = ?, "
            "rescore_model = ? WHERE id = ? AND tenant_id = ?",
            (rescore_score, rescore_reason, rescore_model, candidate_id, tenant_id),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT * FROM candidates WHERE id = ? AND tenant_id = ?",
            (candidate_id, tenant_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise ChalybClipError(f"candidate not found: {candidate_id}")
        return _candidate_from_row(row)


def _candidate_from_row(row: aiosqlite.Row) -> CandidateRow:
    d = dict(row)
    d["evidence"] = json.loads(d.pop("evidence_json") or "{}")
    return CandidateRow.model_validate(d)


class ClipsRepo:
    """Clips. Inserts are INSERT OR IGNORE so the same (id) on resume is a no-op."""

    def __init__(self, db: Database):
        self._db = db

    async def upsert_many(self, rows: list[ClipRow]) -> int:
        if not rows:
            return 0
        bound = current_tenant_id()
        for row in rows:
            if row.tenant_id != bound:
                raise TenancyError(f"clip tenant {row.tenant_id!r} != bound {bound!r}")
        conn = await self._db.connect()
        await conn.executemany(
            "INSERT INTO clips "
            "(id, stream_id, tenant_id, candidate_id, start_s, end_s, duration_s, "
            "width, height, path, smart_crop_box_json, thumbnail_frame_path, status, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [
                (
                    r.id,
                    r.stream_id,
                    r.tenant_id,
                    r.candidate_id,
                    r.start_s,
                    r.end_s,
                    r.duration_s,
                    r.width,
                    r.height,
                    r.path,
                    json.dumps(r.smart_crop_box) if r.smart_crop_box else None,
                    r.thumbnail_frame_path,
                    r.status,
                    r.created_at,
                )
                for r in rows
            ],
        )
        await conn.commit()
        return len(rows)

    async def get(self, clip_id: str) -> ClipRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM clips WHERE id = ? AND tenant_id = ?",
            (clip_id, tenant_id),
        )
        row = await cur.fetchone()
        return _clip_from_row(row) if row else None

    async def list_for_stream(
        self, stream_id: str, *, limit: int = 10_000
    ) -> list[ClipRow]:
        """List clips for a stream, hard-capped to `limit` rows.

        Default 10 000 mirrors CandidatesRepo.list_for_stream; protects
        against the dashboard rendering with an unbounded result set.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM clips WHERE tenant_id = ? AND stream_id = ? "
            "ORDER BY created_at LIMIT ?",
            (tenant_id, stream_id, int(limit)),
        )
        return [_clip_from_row(r) for r in await cur.fetchall()]

    async def list_for_tenant_with_status(
        self, statuses: list[str], *, limit: int = 500
    ) -> list[ClipRow]:
        """Slice O.8 — every clip across every stream with one of the
        given statuses. Powers the global Publish page which aggregates
        approved + published clips across the whole tenant.

        Newest-first because the operator usually wants their latest
        approvals at the top of the list.
        """
        if not statuses:
            return []
        tenant_id = current_tenant_id()
        placeholders = ",".join("?" for _ in statuses)
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT * FROM clips WHERE tenant_id = ? "
            f"AND status IN ({placeholders}) "
            f"ORDER BY created_at DESC LIMIT ?",
            (tenant_id, *statuses, limit),
        )
        return [_clip_from_row(r) for r in await cur.fetchall()]

    async def count_for_tenant_with_status(self, statuses: list[str]) -> int:
        """COUNT of the tenant's clips in the given statuses — the un-capped
        companion to `list_for_tenant_with_status`, for surfaces that show a
        display-limited page of clips but must label actions with the REAL
        total (the Auto-programar button read the 60-row grid's length and
        under-reported any backlog bigger than the page)."""
        if not statuses:
            return 0
        tenant_id = current_tenant_id()
        placeholders = ",".join("?" for _ in statuses)
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT COUNT(*) AS n FROM clips WHERE tenant_id = ? "
            f"AND status IN ({placeholders})",
            (tenant_id, *statuses),
        )
        row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def set_overlay_config(
        self, clip_id: str, *, overlay_config: dict[str, object] | None
    ) -> ClipRow:
        """Persist the per-clip overlay config (set in the clip editor).

        `None` clears the column → renderer falls back to brand-kit
        defaults end-to-end.
        """
        tenant_id = current_tenant_id()
        existing = await self.get(clip_id)
        if existing is None:
            raise ChalybClipError(f"clip {clip_id!r} not found")
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE clips SET overlay_config_json = ? "
            "WHERE id = ? AND tenant_id = ?",
            (
                json.dumps(overlay_config) if overlay_config is not None else None,
                clip_id,
                tenant_id,
            ),
        )
        await conn.commit()
        out = await self.get(clip_id)
        assert out is not None
        return out

    async def set_publishability(
        self,
        clip_id: str,
        *,
        score: int,
        status: str,
    ) -> ClipRow:
        """Slice G.2 — cache the publishability verdict on the clip row.

        Called from the editor render path + every save endpoint so the
        inbox + streams grid can render a status chip without rerunning
        the scorer. `status` is one of the PublishStatus literals
        (publish_ready / needs_edit / reject).
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE clips SET publishability_score = ?, "
            "publishability_status = ? WHERE id = ? AND tenant_id = ?",
            (int(score), status, clip_id, tenant_id),
        )
        await conn.commit()
        out = await self.get(clip_id)
        if out is None:
            raise ChalybClipError(f"clip {clip_id!r} not found")
        return out

    # ------------------------------------------------------------------
    # Render state — Migration T1
    # ------------------------------------------------------------------

    async def mark_render_started(self, clip_id: str) -> None:
        """Atomically transition the clip into the 'rendering' state.

        Called from the download endpoint right before scheduling the
        background render task. The atomic check on (state != 'rendering')
        means a second click on the Download button while the first
        render is in flight is a no-op — there's exactly one background
        task per clip at any time, no thundering herd.
        """
        tenant_id = current_tenant_id()
        from datetime import UTC, datetime
        now_iso = datetime.now(UTC).isoformat()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE clips SET render_state = 'rendering', "
            "render_progress_pct = 0, render_error = NULL, "
            "render_started_at = ? "
            "WHERE id = ? AND tenant_id = ? AND render_state != 'rendering'",
            (now_iso, clip_id, tenant_id),
        )
        await conn.commit()

    async def update_render_progress(
        self, clip_id: str, *, pct: int
    ) -> None:
        """Bump the visible progress percentage without touching state.

        Called by the recorder's capture_progress emitter so the UI
        can show "Rendering 67%..." instead of an opaque spinner.
        Clamps 0-100 defensively.
        """
        tenant_id = current_tenant_id()
        clamped = max(0, min(100, int(pct)))
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE clips SET render_progress_pct = ? "
            "WHERE id = ? AND tenant_id = ? AND render_state = 'rendering'",
            (clamped, clip_id, tenant_id),
        )
        await conn.commit()

    async def mark_render_ready(self, clip_id: str) -> None:
        """Terminal: render finished, MP4 is on disk. Clear any prior
        error from a previous failed attempt."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE clips SET render_state = 'ready', "
            "render_progress_pct = 100, render_error = NULL "
            "WHERE id = ? AND tenant_id = ?",
            (clip_id, tenant_id),
        )
        await conn.commit()

    async def mark_render_failed(
        self, clip_id: str, *, error: str
    ) -> None:
        """Terminal: render blew up. UI surfaces the message + a retry
        button. Error truncated to 300 chars so a ffmpeg essay doesn't
        balloon the column."""
        tenant_id = current_tenant_id()
        truncated = (error or "")[:300]
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE clips SET render_state = 'failed', "
            "render_error = ? "
            "WHERE id = ? AND tenant_id = ?",
            (truncated, clip_id, tenant_id),
        )
        await conn.commit()

    async def reset_render_state(self, clip_id: str) -> None:
        """Reset to 'idle' — used by the overlay-save path (cache
        invalidation) and the "Retry render" button."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE clips SET render_state = 'idle', "
            "render_progress_pct = 0, render_error = NULL, "
            "render_started_at = NULL "
            "WHERE id = ? AND tenant_id = ?",
            (clip_id, tenant_id),
        )
        await conn.commit()

    async def set_trim_bounds(
        self,
        clip_id: str,
        *,
        start_s: float,
        end_s: float,
        duration_s: float,
        new_path: str,
        original_start_s: float | None,
        original_end_s: float | None,
    ) -> ClipRow:
        """Slice G.4b — replace the clip's bounds + path after auto-trim.

        Stores the originals on the FIRST trim so revert is one query.
        Caller is responsible for not overwriting them on a second
        auto-trim (re-pass `existing.original_start_s` if already set).
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE clips SET start_s = ?, end_s = ?, duration_s = ?, "
            "path = ?, original_start_s = ?, original_end_s = ? "
            "WHERE id = ? AND tenant_id = ?",
            (
                float(start_s),
                float(end_s),
                float(duration_s),
                new_path,
                original_start_s,
                original_end_s,
                clip_id,
                tenant_id,
            ),
        )
        await conn.commit()
        out = await self.get(clip_id)
        if out is None:
            raise ChalybClipError(f"clip {clip_id!r} not found")
        return out

    async def clear_trim_bounds(self, clip_id: str) -> ClipRow:
        """Slice G.4b — wipe original_start_s / original_end_s after a
        successful revert. Caller is responsible for restoring
        start_s/end_s/duration_s + path back to the original via
        set_trim_bounds (with originals=None) BEFORE calling this."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE clips SET original_start_s = NULL, "
            "original_end_s = NULL WHERE id = ? AND tenant_id = ?",
            (clip_id, tenant_id),
        )
        await conn.commit()
        out = await self.get(clip_id)
        if out is None:
            raise ChalybClipError(f"clip {clip_id!r} not found")
        return out

    async def update_status(self, clip_id: str, *, status: str) -> ClipRow:
        """Direct status write — caller checks the transition graph.

        Surface for the clip editor's "Complete" button which moves the
        clip to 'approved' (the existing pre-publish standby state) in
        the same request that persists the overlay config.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE clips SET status = ? WHERE id = ? AND tenant_id = ?",
            (status, clip_id, tenant_id),
        )
        await conn.commit()
        out = await self.get(clip_id)
        if out is None:
            raise ChalybClipError(f"clip {clip_id!r} not found")
        return out


def _clip_from_row(row: aiosqlite.Row) -> ClipRow:
    """Deserialize a `clips` row into ClipRow.

    Defensive against schema/model skew: we explicitly translate the
    JSON columns we know about, then filter `d` to only the keys
    ClipRow declares. This means a future column addition doesn't
    crash the page if someone forgets to update this function — the
    new column just gets dropped on the floor here. Trade-off: a bug
    where the model field is misspelled wouldn't surface as a
    validation error any more, but the static type-check + test
    suite catch that earlier.
    """
    d = dict(row)
    box = d.pop("smart_crop_box_json", None)
    d["smart_crop_box"] = json.loads(box) if box else None
    overlay = d.pop("overlay_config_json", None)
    d["overlay_config"] = json.loads(overlay) if overlay else None
    known = set(ClipRow.model_fields.keys())
    filtered = {k: v for k, v in d.items() if k in known}
    return ClipRow.model_validate(filtered)


class VariantsRepo:
    """Variants per (clip, persona). Re-running with a new persona adds rows."""

    def __init__(self, db: Database):
        self._db = db

    async def replace_for_clip_persona(
        self, clip_id: str, persona_id: str, rows: list[VariantRow]
    ) -> int:
        """Delete the existing batch for (clip, persona), then insert the new rows."""
        bound = current_tenant_id()
        for row in rows:
            if row.tenant_id != bound:
                raise TenancyError(f"variant tenant {row.tenant_id!r} != bound {bound!r}")
            if row.clip_id != clip_id or row.persona_id != persona_id:
                raise TenancyError("variant clip_id/persona_id must match the replace target")
        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM variants WHERE tenant_id = ? AND clip_id = ? AND persona_id = ?",
            (bound, clip_id, persona_id),
        )
        if rows:
            await conn.executemany(
                "INSERT INTO variants "
                "(id, clip_id, tenant_id, persona_id, language, caption, "
                "title_card_text, hashtags_json, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        r.id,
                        r.clip_id,
                        r.tenant_id,
                        r.persona_id,
                        r.language,
                        r.caption,
                        r.title_card_text,
                        json.dumps(r.hashtags),
                        r.model,
                        r.created_at,
                    )
                    for r in rows
                ],
            )
        await conn.commit()
        return len(rows)

    async def list_for_clip(
        self, clip_id: str, *, persona_id: str | None = None
    ) -> list[VariantRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        if persona_id is None:
            cur = await conn.execute(
                "SELECT * FROM variants WHERE tenant_id = ? AND clip_id = ? ORDER BY created_at",
                (tenant_id, clip_id),
            )
        else:
            cur = await conn.execute(
                "SELECT * FROM variants WHERE tenant_id = ? AND clip_id = ? "
                "AND persona_id = ? ORDER BY created_at",
                (tenant_id, clip_id, persona_id),
            )
        return [_variant_from_row(r) for r in await cur.fetchall()]


def _variant_from_row(row: aiosqlite.Row) -> VariantRow:
    d = dict(row)
    d["hashtags"] = json.loads(d.pop("hashtags_json") or "[]")
    return VariantRow.model_validate(d)


_ACCOUNT_COLS = (
    "id, tenant_id, platform, external_id, display_name, oauth_blob_json, "
    "created_at, refresh_token, expires_at, scopes_json, status, "
    # Migration 021 columns kept in the SELECT so prod rows that
    # touched them validate cleanly. No active code reads these
    # any more — the in-house Connect flow was scrapped in favor
    # of upload-post.
    "access_token_encrypted, refresh_token_encrypted, token_type, "
    "platform_user_id, platform_username, platform_avatar_url, "
    "daily_publish_count, daily_publish_window_start"
)


class ConnectedAccountsRepo:
    """Per-tenant social-account connections (Buffer / TikTok / YouTube).

    Phase 2 adds OAuth refresh tracking (`refresh_token`, `expires_at`,
    `scopes`, `status`). The publisher dispatcher (P2 Task 10) refreshes
    expired tokens via `update_oauth` before posting.
    """

    def __init__(self, db: Database):
        self._db = db

    async def create(
        self,
        *,
        platform: str,
        external_id: str,
        display_name: str | None = None,
        oauth_blob: dict[str, object] | None = None,
        refresh_token: str | None = None,
        expires_at: str | None = None,
        scopes: list[str] | None = None,
    ) -> ConnectedAccount:
        tenant_id = current_tenant_id()
        account_id = new_id("acc")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO connected_accounts "
            "(id, tenant_id, platform, external_id, display_name, oauth_blob_json, "
            "created_at, refresh_token, expires_at, scopes_json, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')",
            (
                account_id,
                tenant_id,
                platform,
                external_id,
                display_name,
                json.dumps(oauth_blob) if oauth_blob is not None else None,
                _now(),
                refresh_token,
                expires_at,
                json.dumps(scopes) if scopes is not None else None,
            ),
        )
        await conn.commit()
        acct = await self.get(account_id)
        assert acct is not None
        return acct

    async def get(self, account_id: str) -> ConnectedAccount | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_ACCOUNT_COLS} FROM connected_accounts "
            "WHERE id = ? AND tenant_id = ?",
            (account_id, tenant_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return _connected_account_from_row(row)

    async def list_for_tenant(self) -> list[ConnectedAccount]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_ACCOUNT_COLS} FROM connected_accounts "
            "WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        )
        return [_connected_account_from_row(r) for r in await cur.fetchall()]

    async def update_oauth(
        self,
        account_id: str,
        *,
        oauth_blob: dict[str, object] | None = None,
        refresh_token: str | None = None,
        expires_at: str | None = None,
        scopes: list[str] | None = None,
    ) -> ConnectedAccount:
        """Persist a fresh OAuth bundle after a refresh round-trip.

        Only the kwargs you pass get updated; pass None to leave a column
        alone. Status stays as-is unless `mark_status` is called separately.
        """
        tenant_id = current_tenant_id()
        sets: list[str] = []
        values: list[object] = []
        if oauth_blob is not None:
            sets.append("oauth_blob_json = ?")
            values.append(json.dumps(oauth_blob))
        if refresh_token is not None:
            sets.append("refresh_token = ?")
            values.append(refresh_token)
        if expires_at is not None:
            sets.append("expires_at = ?")
            values.append(expires_at)
        if scopes is not None:
            sets.append("scopes_json = ?")
            values.append(json.dumps(scopes))
        if not sets:
            existing = await self.get(account_id)
            if existing is None:
                raise ChalybClipError(f"connected_account not found: {account_id}")
            return existing
        values.extend([account_id, tenant_id])
        conn = await self._db.connect()
        await conn.execute(
            f"UPDATE connected_accounts SET {', '.join(sets)} "
            "WHERE id = ? AND tenant_id = ?",
            tuple(values),
        )
        await conn.commit()
        existing = await self.get(account_id)
        if existing is None:
            raise ChalybClipError(f"connected_account not found: {account_id}")
        return existing

    async def mark_status(self, account_id: str, status: str) -> ConnectedAccount:
        """Flip an account's lifecycle status (`active` / `auth_failed` / `disabled`)."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE connected_accounts SET status = ? WHERE id = ? AND tenant_id = ?",
            (status, account_id, tenant_id),
        )
        await conn.commit()
        existing = await self.get(account_id)
        if existing is None:
            raise ChalybClipError(f"connected_account not found: {account_id}")
        return existing

    async def update_meta(
        self,
        account_id: str,
        *,
        display_name: str | None = None,
        external_id: str | None = None,
    ) -> ConnectedAccount:
        """Slice O.7 — patch the operator-editable fields on a connection.

        Display-name + external-id are the two things an operator might
        want to tweak from the UI without doing a full re-OAuth (e.g. they
        renamed their account, or they originally typed a typo in the
        IG-Business ID). Pass None to leave the column alone.
        """
        tenant_id = current_tenant_id()
        sets: list[str] = []
        values: list[object] = []
        if display_name is not None:
            sets.append("display_name = ?")
            values.append(display_name or None)
        if external_id is not None:
            sets.append("external_id = ?")
            values.append(external_id)
        if not sets:
            existing = await self.get(account_id)
            if existing is None:
                raise ChalybClipError(f"connected_account not found: {account_id}")
            return existing
        values.extend([account_id, tenant_id])
        conn = await self._db.connect()
        await conn.execute(
            f"UPDATE connected_accounts SET {', '.join(sets)} "
            "WHERE id = ? AND tenant_id = ?",
            tuple(values),
        )
        await conn.commit()
        existing = await self.get(account_id)
        if existing is None:
            raise ChalybClipError(f"connected_account not found: {account_id}")
        return existing


def _connected_account_from_row(row: aiosqlite.Row) -> ConnectedAccount:
    d = dict(row)
    blob = d.pop("oauth_blob_json")
    d["oauth_blob"] = json.loads(blob) if blob else None
    scopes_blob = d.pop("scopes_json", None)
    d["scopes"] = json.loads(scopes_blob) if scopes_blob else []
    # Migration 021 columns flow through as-is — Pydantic accepts the
    # encrypted blobs as bytes and the rest as str/int. Older rows
    # (pre-migration) have NULL for these and the model's defaults
    # kick in.
    return ConnectedAccount.model_validate(d)


class PublishJobsRepo:
    """Pending / running / sent publish jobs.

    Task 9 uses `enqueue` to write rows; the worker (Phase 1 Buffer / P2
    TikTok+YT) reads pending jobs via `list_pending`, transitions them
    through the status column, and records final state.

    Phase 2 adds `external_url` + `platform_metadata_json` for native-
    publisher response payloads (the URL we'd open in a browser, plus
    raw API bits we want to keep around for debugging).
    """

    def __init__(self, db: Database):
        self._db = db

    async def enqueue(
        self,
        *,
        clip_id: str,
        variant_id: str,
        account_id: str,
        platform: str,
        scheduled_for: str | None = None,
        platform_metadata: dict | None = None,
    ) -> PublishJob:
        # Slice O.6 — operator-supplied per-platform metadata
        # (title, description, hashtags, platform-specific knobs like
        # YT category or IG-Reels cover-frame). The worker reads this
        # blob and merges it onto the platform API call.
        import json as _json

        tenant_id = current_tenant_id()
        job_id = new_id("pjb")
        meta_json = _json.dumps(platform_metadata) if platform_metadata else None
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO publish_jobs "
            "(id, tenant_id, clip_id, variant_id, account_id, platform, "
            "status, attempts, last_error, scheduled_for, external_id, "
            "external_url, platform_metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, NULL, ?, NULL, NULL, ?, ?)",
            (
                job_id,
                tenant_id,
                clip_id,
                variant_id,
                account_id,
                platform,
                scheduled_for,
                meta_json,
                _now(),
            ),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT * FROM publish_jobs WHERE id = ? AND tenant_id = ?",
            (job_id, tenant_id),
        )
        row = await cur.fetchone()
        assert row is not None
        return _publish_job_from_row(row)

    async def list_for_clip(self, clip_id: str) -> list[PublishJob]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM publish_jobs WHERE tenant_id = ? AND clip_id = ? "
            "ORDER BY created_at",
            (tenant_id, clip_id),
        )
        return [_publish_job_from_row(r) for r in await cur.fetchall()]

    async def list_pending(self, *, limit: int = 50) -> list[PublishJob]:
        """Pending jobs whose `scheduled_for` window has elapsed.

        A job stays invisible to the worker while `scheduled_for` is in
        the future — that's how the auto-publish undo window (slice E.2)
        works: enqueue with `scheduled_for = now + delay_min`, then the
        operator has that long to cancel before the worker picks it up.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM publish_jobs WHERE tenant_id = ? AND status = 'pending' "
            "AND (scheduled_for IS NULL OR scheduled_for <= ?) "
            "ORDER BY created_at LIMIT ?",
            (tenant_id, _now(), limit),
        )
        return [_publish_job_from_row(r) for r in await cur.fetchall()]

    async def list_scheduled(self, *, limit: int = 50) -> list[PublishJob]:
        """Jobs sitting in the undo window — pending AND scheduled_for is
        in the future. Surfaced on the dashboard so the operator can
        cancel before the worker picks them up."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM publish_jobs WHERE tenant_id = ? AND status = 'pending' "
            "AND scheduled_for IS NOT NULL AND scheduled_for > ? "
            "ORDER BY scheduled_for LIMIT ?",
            (tenant_id, _now(), limit),
        )
        return [_publish_job_from_row(r) for r in await cur.fetchall()]

    async def cancel(self, job_id: str) -> bool:
        """Mark a pending job as canceled (the 'undo' button).

        Returns True iff the job was pending AND owned by the current
        tenant. Sent/failed jobs are NOT cancelable — at that point the
        platform call already fired.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "UPDATE publish_jobs SET status = 'canceled', last_error = NULL "
            "WHERE id = ? AND tenant_id = ? AND status = 'pending'",
            (job_id, tenant_id),
        )
        await conn.commit()
        return bool(cur.rowcount)

    async def retry(self, job_id: str) -> bool:
        """Slice O.7 — flip a failed job back to `pending` so the worker
        picks it up on the next pass.

        Returns True iff the job was failed AND owned by the current
        tenant. Doesn't reset the `attempts` counter — that's intentional
        so we can see how many times a job has been retried in the
        history.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "UPDATE publish_jobs SET status = 'pending', last_error = NULL "
            "WHERE id = ? AND tenant_id = ? AND status = 'failed'",
            (job_id, tenant_id),
        )
        await conn.commit()
        return bool(cur.rowcount)

    async def list_recent_for_tenant(
        self, *, status: str | None = None, limit: int = 50
    ) -> list[PublishJob]:
        """Recent jobs for the bound tenant, newest first.

        `status=None` returns every status; pass "pending" / "sent" /
        "failed" to scope. Used by the `chalybclip queue list` CLI view.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        if status is None:
            cur = await conn.execute(
                "SELECT * FROM publish_jobs WHERE tenant_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (tenant_id, limit),
            )
        else:
            cur = await conn.execute(
                "SELECT * FROM publish_jobs WHERE tenant_id = ? AND status = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (tenant_id, status, limit),
            )
        return [_publish_job_from_row(r) for r in await cur.fetchall()]

    async def reschedule(self, job_id: str, *, scheduled_for: str) -> bool:
        """Push a still-pending job's `scheduled_for` forward.

        Used by the safe trap (safe-window gate) to defer a job that would
        post inside a blocked window — it reappears to the worker once the
        new `scheduled_for` elapses. Only pending jobs move; sent/failed
        jobs are untouched.

        Returns True iff a pending row owned by the current tenant moved.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "UPDATE publish_jobs SET scheduled_for = ? "
            "WHERE id = ? AND tenant_id = ? AND status = 'pending'",
            (scheduled_for, job_id, tenant_id),
        )
        await conn.commit()
        return bool(cur.rowcount)

    async def recent_post_times(
        self,
        *,
        platform: str,
        since: str,
        exclude_job_id: str | None = None,
    ) -> list[str]:
        """Effective post times for `platform` since `since` (ISO, UTC).

        Drives the safe trap's spacing + daily-cap math. We count jobs that
        already shipped (`sent`) or are still in flight (`pending`, incl.
        future-scheduled), using `COALESCE(scheduled_for, created_at)` as
        the moment the post lands. `canceled` / `failed` jobs don't count —
        they never reach (or no longer occupy) the platform.

        `exclude_job_id` drops the job being evaluated so it doesn't collide
        with its own scheduled slot.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        sql = (
            "SELECT COALESCE(scheduled_for, created_at) AS t FROM publish_jobs "
            "WHERE tenant_id = ? AND platform = ? "
            "AND status IN ('sent', 'pending') "
            "AND COALESCE(scheduled_for, created_at) >= ?"
        )
        params: list[object] = [tenant_id, platform, since]
        if exclude_job_id is not None:
            sql += " AND id != ?"
            params.append(exclude_job_id)
        cur = await conn.execute(sql, tuple(params))
        return [str(r[0]) for r in await cur.fetchall()]

    async def count_for_tenant_today(self, *, platform: str | None = None) -> int:
        """Today's (UTC) publish_jobs count for the bound tenant.

        The budget governor (P2 Task 1) calls this to enforce
        `tenants.daily_publish_limit`. `platform=None` counts across all
        platforms; passing a value scopes it.
        """
        tenant_id = current_tenant_id()
        cutoff = _start_of_utc_today()
        conn = await self._db.connect()
        if platform is None:
            cur = await conn.execute(
                "SELECT COUNT(*) FROM publish_jobs "
                "WHERE tenant_id = ? AND created_at >= ?",
                (tenant_id, cutoff),
            )
        else:
            cur = await conn.execute(
                "SELECT COUNT(*) FROM publish_jobs "
                "WHERE tenant_id = ? AND platform = ? AND created_at >= ?",
                (tenant_id, platform, cutoff),
            )
        row = await cur.fetchone()
        return int(row[0]) if row else 0


def _publish_job_from_row(row: aiosqlite.Row) -> PublishJob:
    d = dict(row)
    meta = d.pop("platform_metadata_json", None)
    d["platform_metadata"] = json.loads(meta) if meta else None
    return PublishJob.model_validate(d)


def _start_of_utc_today() -> str:
    """ISO-8601 timestamp for 00:00:00 UTC today (used as a 'since' filter)."""
    now = _dt.datetime.now(_dt.UTC)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


class VisualSignalsRepo:
    """One row per second of visual signals per stream.

    `replace_for_stream` is the canonical write — re-running analyze_video
    wipes the prior batch and inserts the new one. Composite PK is
    `(stream_id, ts_offset_s)`.
    """

    def __init__(self, db: Database):
        self._db = db

    async def replace_for_stream(
        self,
        stream_id: str,
        track: object,  # chalybclip.vision.VisualSignalTrack — avoid import cycle
    ) -> int:
        bound = current_tenant_id()
        # Late-bound to avoid a hard import cycle with chalybclip.vision.
        from chalybclip.vision import VisualSignalTrack

        if not isinstance(track, VisualSignalTrack):
            raise TenancyError("replace_for_stream requires a VisualSignalTrack")
        if track.tenant_id != bound:
            raise TenancyError(f"visual_signals tenant {track.tenant_id!r} != bound {bound!r}")
        if track.stream_id != stream_id:
            raise TenancyError(f"track.stream_id {track.stream_id!r} != target {stream_id!r}")
        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM visual_signals WHERE tenant_id = ? AND stream_id = ?",
            (bound, stream_id),
        )
        if track.signals:
            await conn.executemany(
                "INSERT INTO visual_signals "
                "(stream_id, tenant_id, ts_offset_s, scene_cut, face_emotion, "
                "motion_energy, text_changed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        stream_id,
                        bound,
                        s.ts_offset_s,
                        1 if s.scene_cut else 0,
                        s.face_emotion,
                        s.motion_energy,
                        1 if s.text_changed else 0,
                    )
                    for s in track.signals
                ],
            )
        await conn.commit()
        return len(track.signals)

    async def list_for_stream(self, stream_id: str) -> list[dict[str, Any]]:
        """Return raw rows so callers (or tests) can compare directly."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT stream_id, tenant_id, ts_offset_s, scene_cut, face_emotion, "
            "motion_energy, text_changed FROM visual_signals "
            "WHERE tenant_id = ? AND stream_id = ? ORDER BY ts_offset_s",
            (tenant_id, stream_id),
        )
        rows = await cur.fetchall()
        return [
            {
                "stream_id": r["stream_id"],
                "tenant_id": r["tenant_id"],
                "ts_offset_s": float(r["ts_offset_s"]),
                "scene_cut": bool(r["scene_cut"]),
                "face_emotion": r["face_emotion"],
                "motion_energy": r["motion_energy"],
                "text_changed": bool(r["text_changed"]),
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Webhook subscriptions (P2 Task 12 worker drains; this repo is the storage).
# ---------------------------------------------------------------------------


_WEBHOOK_COLS = (
    "id, tenant_id, url, types_json, secret, status, created_at, "
    "last_dispatch_ts, failure_count"
)


class WebhookSubscriptionsRepo:
    """CRUD over `webhook_subscriptions`.

    Phase 2 stores one HMAC secret per subscription, generated at create
    time and returned to the caller once (mirrors `api_tokens`).
    """

    def __init__(self, db: Database):
        self._db = db

    async def create(
        self,
        *,
        url: str,
        types: list[str],
        secret: str,
    ) -> WebhookSubscription:
        tenant_id = current_tenant_id()
        sub_id = new_id("whk")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO webhook_subscriptions "
            "(id, tenant_id, url, types_json, secret, status, created_at, "
            "last_dispatch_ts, failure_count) "
            "VALUES (?, ?, ?, ?, ?, 'active', ?, NULL, 0)",
            (sub_id, tenant_id, url, json.dumps(types), secret, _now()),
        )
        await conn.commit()
        sub = await self.get(sub_id)
        assert sub is not None
        return sub

    async def get(self, sub_id: str) -> WebhookSubscription | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_WEBHOOK_COLS} FROM webhook_subscriptions "
            "WHERE id = ? AND tenant_id = ?",
            (sub_id, tenant_id),
        )
        row = await cur.fetchone()
        return _webhook_from_row(row) if row else None

    async def list_for_tenant(
        self, *, status: str | None = None
    ) -> list[WebhookSubscription]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        if status is None:
            cur = await conn.execute(
                f"SELECT {_WEBHOOK_COLS} FROM webhook_subscriptions "
                "WHERE tenant_id = ? ORDER BY created_at",
                (tenant_id,),
            )
        else:
            cur = await conn.execute(
                f"SELECT {_WEBHOOK_COLS} FROM webhook_subscriptions "
                "WHERE tenant_id = ? AND status = ? ORDER BY created_at",
                (tenant_id, status),
            )
        return [_webhook_from_row(r) for r in await cur.fetchall()]

    async def delete(self, sub_id: str) -> bool:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "DELETE FROM webhook_subscriptions WHERE id = ? AND tenant_id = ?",
            (sub_id, tenant_id),
        )
        await conn.commit()
        return (cur.rowcount or 0) > 0

    async def record_dispatch(self, sub_id: str, *, ts: str) -> None:
        """Bump `last_dispatch_ts` after a successful drain pass."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE webhook_subscriptions SET last_dispatch_ts = ?, failure_count = 0 "
            "WHERE id = ? AND tenant_id = ?",
            (ts, sub_id, tenant_id),
        )
        await conn.commit()

    async def record_failure(self, sub_id: str) -> int:
        """Increment `failure_count` after a failed POST. Returns the new count."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE webhook_subscriptions SET failure_count = failure_count + 1 "
            "WHERE id = ? AND tenant_id = ?",
            (sub_id, tenant_id),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT failure_count FROM webhook_subscriptions "
            "WHERE id = ? AND tenant_id = ?",
            (sub_id, tenant_id),
        )
        row = await cur.fetchone()
        return int(row[0]) if row else 0


def _webhook_from_row(row: aiosqlite.Row) -> WebhookSubscription:
    d = dict(row)
    d["types"] = json.loads(d.pop("types_json") or "[]")
    return WebhookSubscription.model_validate(d)


class WebhookSecretsRepo:
    """Phase 3: secret-rotation history for one subscription.

    Each `rotate(...)` writes the current secret to `webhook_secret_versions`
    with an `expires_at` grace deadline, then mints a new one and stamps it
    onto `webhook_subscriptions.secret`. Subscribers can list active
    (unexpired) secrets to verify HMAC signatures against either the
    current or a prior secret during the rotation window.
    """

    def __init__(self, db: Database):
        self._db = db

    async def rotate(
        self, subscription_id: str, *, new_secret: str, ttl_s: float
    ) -> str:
        """Rotate the subscription's secret. Returns the *new* secret.

        Saves the previous secret to `webhook_secret_versions` with
        `expires_at = now + ttl_s`. Caller is responsible for mint quality
        (use `secrets.token_hex(32)`).
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT secret FROM webhook_subscriptions "
            "WHERE id = ? AND tenant_id = ?",
            (subscription_id, tenant_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise ChalybClipError(f"webhook subscription not found: {subscription_id}")
        prior_secret = str(row["secret"])
        if not prior_secret:
            raise ChalybClipError(
                f"webhook subscription {subscription_id} has no current secret"
            )

        version_id = new_id("whk")
        expires_at = (
            _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=ttl_s)
        ).isoformat()
        await conn.execute(
            "INSERT INTO webhook_secret_versions "
            "(id, subscription_id, tenant_id, secret, expires_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (version_id, subscription_id, tenant_id, prior_secret, expires_at, _now()),
        )
        await conn.execute(
            "UPDATE webhook_subscriptions SET secret = ? "
            "WHERE id = ? AND tenant_id = ?",
            (new_secret, subscription_id, tenant_id),
        )
        await conn.commit()
        return new_secret

    async def list_active_for_subscription(
        self, subscription_id: str
    ) -> list[WebhookSecretVersion]:
        """Past secrets whose grace window has not elapsed (newest first)."""
        tenant_id = current_tenant_id()
        now_iso = _now()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT id, subscription_id, tenant_id, secret, expires_at, created_at "
            "FROM webhook_secret_versions "
            "WHERE tenant_id = ? AND subscription_id = ? AND expires_at >= ? "
            "ORDER BY created_at DESC",
            (tenant_id, subscription_id, now_iso),
        )
        return [
            WebhookSecretVersion.model_validate(dict(r)) for r in await cur.fetchall()
        ]

    async def purge_expired(self) -> int:
        """Delete past-grace secrets. Returns the row count deleted.

        Phase 3 polish: a periodic worker can call this hourly. For now,
        the dashboard's "list active secrets" call ignores expired rows
        anyway, so the table just grows linearly with rotations.
        """
        tenant_id = current_tenant_id()
        now_iso = _now()
        conn = await self._db.connect()
        cur = await conn.execute(
            "DELETE FROM webhook_secret_versions "
            "WHERE tenant_id = ? AND expires_at < ?",
            (tenant_id, now_iso),
        )
        await conn.commit()
        return cur.rowcount or 0


# ---------------------------------------------------------------------------
# Phase 3: publish_metrics — engagement-stats snapshots per (job, fetch).
# ---------------------------------------------------------------------------


_METRIC_COLS = (
    "id, tenant_id, publish_job_id, platform, fetched_at, "
    "views, likes, comments, shares, retention_pct, ctr, "
    "raw_metadata_json, created_at"
)


class PublishMetricsRepo:
    """Append-only snapshots of platform engagement stats per publish_job.

    The dashboard's outcome card reads the latest row per job; the
    calibration loop reads the time series. UPDATEs are never used -
    each fetch writes a new row.
    """

    def __init__(self, db: Database):
        self._db = db

    async def record(
        self,
        *,
        publish_job_id: str,
        platform: str,
        fetched_at: str,
        views: int | None = None,
        likes: int | None = None,
        comments: int | None = None,
        shares: int | None = None,
        retention_pct: float | None = None,
        ctr: float | None = None,
        raw_metadata: dict[str, object] | None = None,
    ) -> PublishMetric:
        tenant_id = current_tenant_id()
        metric_id = new_id("met")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO publish_metrics "
            "(id, tenant_id, publish_job_id, platform, fetched_at, "
            "views, likes, comments, shares, retention_pct, ctr, "
            "raw_metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                metric_id,
                tenant_id,
                publish_job_id,
                platform,
                fetched_at,
                views,
                likes,
                comments,
                shares,
                retention_pct,
                ctr,
                json.dumps(raw_metadata) if raw_metadata is not None else None,
                _now(),
            ),
        )
        await conn.commit()
        cur = await conn.execute(
            f"SELECT {_METRIC_COLS} FROM publish_metrics WHERE id = ? AND tenant_id = ?",
            (metric_id, tenant_id),
        )
        row = await cur.fetchone()
        assert row is not None
        return _metric_from_row(row)

    async def latest_for_job(self, publish_job_id: str) -> PublishMetric | None:
        """The most recent metric reading for one publish_job."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_METRIC_COLS} FROM publish_metrics "
            "WHERE tenant_id = ? AND publish_job_id = ? "
            "ORDER BY fetched_at DESC LIMIT 1",
            (tenant_id, publish_job_id),
        )
        row = await cur.fetchone()
        return _metric_from_row(row) if row else None

    async def list_for_job(
        self, publish_job_id: str, *, limit: int = 50
    ) -> list[PublishMetric]:
        """Time series for one publish_job, oldest first (chronological)."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_METRIC_COLS} FROM publish_metrics "
            "WHERE tenant_id = ? AND publish_job_id = ? "
            "ORDER BY fetched_at ASC LIMIT ?",
            (tenant_id, publish_job_id, limit),
        )
        return [_metric_from_row(r) for r in await cur.fetchall()]

    async def latest_per_job_since(
        self, *, platform: str, since: str, limit: int = 1000
    ) -> list[PublishMetric]:
        """Latest metric reading per publish_job for a platform since `since`.

        Used by the calibration loop: pair the rescore_score on the source
        candidate with the platform's eventual view count.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        # Fetch every reading since `since` and let the caller take the
        # latest per job (SQLite's lack of DISTINCT ON makes the row-level
        # version of this awkward; the DB is small enough that pulling all
        # rows + dict-collapse here is fine).
        cur = await conn.execute(
            f"SELECT {_METRIC_COLS} FROM publish_metrics "
            "WHERE tenant_id = ? AND platform = ? AND fetched_at >= ? "
            "ORDER BY fetched_at DESC LIMIT ?",
            (tenant_id, platform, since, limit),
        )
        rows = await cur.fetchall()
        # Collapse to the latest row per publish_job_id.
        latest: dict[str, PublishMetric] = {}
        for r in rows:
            metric = _metric_from_row(r)
            if metric.publish_job_id not in latest:
                latest[metric.publish_job_id] = metric
        return list(latest.values())


def _metric_from_row(row: aiosqlite.Row) -> PublishMetric:
    d = dict(row)
    raw = d.pop("raw_metadata_json", None)
    d["raw_metadata"] = json.loads(raw) if raw else None
    return PublishMetric.model_validate(d)


# ---------- Speakers (voice-markers spec slice B.2) ----------


def _speaker_from_row(row: aiosqlite.Row) -> SpeakerRow:
    d = dict(row)
    raw = d.pop("embedding_json", None)
    d["embedding"] = json.loads(raw) if raw else None
    d["is_self"] = bool(d["is_self"])
    return SpeakerRow.model_validate(d)


def _vod_speaker_from_row(row: aiosqlite.Row) -> VodSpeakerRow:
    d = dict(row)
    raw = d.pop("embedding_json", None)
    d["embedding"] = json.loads(raw) if raw else None
    return VodSpeakerRow.model_validate(d)


class SpeakersRepo:
    """Persistent voice identities for a tenant.

    Embedding vectors are stored as JSON list-of-floats (portable to
    Postgres without pgvector — when scale demands it, swap in a real
    vector column with no API change here).
    """

    def __init__(self, db: Database):
        self._db = db

    async def create(
        self,
        *,
        display_name: str,
        is_self: bool = False,
        preferred_brand_kit_id: str | None = None,
        embedding: list[float] | None = None,
        total_speech_s: float = 0.0,
        sample_audio_path: str | None = None,
    ) -> SpeakerRow:
        tenant_id = current_tenant_id()
        speaker_id = new_id("spk")
        now = _now()
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO speakers (id, tenant_id, display_name, is_self, "
            "preferred_brand_kit_id, embedding_json, embedding_dim, "
            "total_speech_s, sample_audio_path, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                speaker_id,
                tenant_id,
                display_name,
                1 if is_self else 0,
                preferred_brand_kit_id,
                json.dumps(embedding) if embedding is not None else None,
                len(embedding) if embedding is not None else None,
                total_speech_s,
                sample_audio_path,
                now,
                now,
            ),
        )
        await conn.commit()
        out = await self.get(speaker_id)
        assert out is not None
        return out

    async def get(self, speaker_id: str) -> SpeakerRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM speakers WHERE id = ? AND tenant_id = ?",
            (speaker_id, tenant_id),
        )
        row = await cur.fetchone()
        return _speaker_from_row(row) if row else None

    async def list_for_tenant(self) -> list[SpeakerRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM speakers WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        )
        return [_speaker_from_row(r) for r in await cur.fetchall()]

    async def update_embedding(
        self,
        *,
        speaker_id: str,
        embedding: list[float],
        total_speech_s: float,
    ) -> None:
        """Fold a new VOD's embedding into the persistent identity.

        Caller is responsible for the merge math (typically a duration-
        weighted average). This method just persists the result.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE speakers SET embedding_json = ?, embedding_dim = ?, "
            "total_speech_s = ?, updated_at = ? "
            "WHERE id = ? AND tenant_id = ?",
            (
                json.dumps(embedding),
                len(embedding),
                total_speech_s,
                _now(),
                speaker_id,
                tenant_id,
            ),
        )
        await conn.commit()

    async def set_display_name(self, speaker_id: str, display_name: str) -> None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE speakers SET display_name = ?, updated_at = ? "
            "WHERE id = ? AND tenant_id = ?",
            (display_name, _now(), speaker_id, tenant_id),
        )
        await conn.commit()

    async def set_preferred_brand_kit(
        self, speaker_id: str, brand_kit_id: str | None
    ) -> SpeakerRow | None:
        """Assign a brand kit to a speaker. Pass `None` to clear.

        With the FK landed by migration 006, an invalid kit_id raises
        a foreign-key error from SQLite — caller should validate the
        kit exists for the tenant before calling.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE speakers SET preferred_brand_kit_id = ?, updated_at = ? "
            "WHERE id = ? AND tenant_id = ?",
            (brand_kit_id, _now(), speaker_id, tenant_id),
        )
        await conn.commit()
        return await self.get(speaker_id)


class VodSpeakersRepo:
    """One row per (stream_id, within-VOD speaker_label).

    Written after each diarization run. The resolved_speaker_id link is
    filled in by the embedding-match step in the pipeline; until that
    runs (or when the user merges/labels later), it stays None.
    """

    def __init__(self, db: Database):
        self._db = db

    async def upsert(
        self,
        *,
        stream_id: str,
        speaker_label: str,
        resolved_speaker_id: str | None,
        confidence: float | None,
        total_speech_s: float,
        embedding: list[float] | None,
    ) -> VodSpeakerRow:
        tenant_id = current_tenant_id()
        row_id = new_id("vsp")
        now = _now()
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO vod_speakers (id, stream_id, tenant_id, speaker_label, "
            "resolved_speaker_id, confidence, total_speech_s, embedding_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(stream_id, speaker_label) DO UPDATE SET "
            "resolved_speaker_id = excluded.resolved_speaker_id, "
            "confidence = excluded.confidence, "
            "total_speech_s = excluded.total_speech_s, "
            "embedding_json = excluded.embedding_json",
            (
                row_id,
                stream_id,
                tenant_id,
                speaker_label,
                resolved_speaker_id,
                confidence,
                total_speech_s,
                json.dumps(embedding) if embedding is not None else None,
                now,
            ),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT * FROM vod_speakers "
            "WHERE stream_id = ? AND speaker_label = ? AND tenant_id = ?",
            (stream_id, speaker_label, tenant_id),
        )
        row = await cur.fetchone()
        assert row is not None
        return _vod_speaker_from_row(row)

    async def list_for_stream(self, stream_id: str) -> list[VodSpeakerRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM vod_speakers WHERE stream_id = ? AND tenant_id = ? "
            "ORDER BY total_speech_s DESC",
            (stream_id, tenant_id),
        )
        return [_vod_speaker_from_row(r) for r in await cur.fetchall()]

    async def list_unresolved_for_tenant(self) -> list[VodSpeakerRow]:
        """Pending-labeling queue for the dashboard's /speakers admin UI."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT * FROM vod_speakers WHERE tenant_id = ? "
            "AND resolved_speaker_id IS NULL ORDER BY total_speech_s DESC",
            (tenant_id,),
        )
        return [_vod_speaker_from_row(r) for r in await cur.fetchall()]


# ---------- Brand kits (voice-markers spec slice C.1) ----------


_BRAND_KIT_COLS = (
    "id, tenant_id, name, is_default, "
    "primary_color, accent_color, text_color, font_family, font_weight, "
    "logo_url, logo_dark_url, watermark_url, intro_sting_url, outro_sting_url, "
    "caption_style_json, default_layout, "
    "handle_tiktok, handle_youtube, handle_instagram, handle_kick, "
    "ai_generated, ai_prompt, ai_provider, "
    "auto_publish_enabled, auto_publish_platforms_json, auto_publish_delay_min, "
    "custom_trigger_phrases_json, "
    # Slice H.1 — user-level editor prefs (migration 010).
    "default_platform, banner_enabled_default, "
    "banner_show_context_default, banner_show_safezones_default, "
    # Slice I.1 — clip style preset + Kick banner variant + top hook
    # (migration 011).
    "clip_style, bottom_banner_style, banner_live_badge_default, "
    "top_hook_enabled_default, top_hook_style_default, "
    # Slice K.5 — operator's default target platform (migration 012).
    "target_platform, "
    # Slice O.1 — pro-tier "show chalybclip credit" toggle (migration 013).
    "show_chalybclip_credit, "
    # Pro-tier "show chalybclip end-card outro" toggle (migration 050).
    "show_chalybclip_outro, "
    # Publishing safe trap (migration 042).
    "safe_schedule_enabled, safety_policy_json, content_timezone, "
    "created_at, updated_at"
)


def _brand_kit_from_row(row: aiosqlite.Row) -> BrandKitRow:
    d = dict(row)
    d["is_default"] = bool(d["is_default"])
    d["ai_generated"] = bool(d["ai_generated"])
    d["auto_publish_enabled"] = bool(d["auto_publish_enabled"])
    # Slice H.1 — new boolean cols from migration 010.
    d["banner_enabled_default"] = bool(d.get("banner_enabled_default", 0))
    d["banner_show_context_default"] = bool(d.get("banner_show_context_default", 0))
    d["banner_show_safezones_default"] = bool(d.get("banner_show_safezones_default", 0))
    # Slice I.1 — new boolean cols from migration 011.
    d["banner_live_badge_default"] = bool(d.get("banner_live_badge_default", 0))
    d["top_hook_enabled_default"] = bool(d.get("top_hook_enabled_default", 0))
    # Slice O.1 — pro-tier credit toggle (migration 013).
    d["show_chalybclip_credit"] = bool(d.get("show_chalybclip_credit", 1))
    # Pro-tier end-card outro toggle (migration 050).
    d["show_chalybclip_outro"] = bool(d.get("show_chalybclip_outro", 1))
    # Safe trap (migration 042).
    d["safe_schedule_enabled"] = bool(d.get("safe_schedule_enabled", 0))
    raw_safety = d.pop("safety_policy_json", None)
    d["safety_policy"] = json.loads(raw_safety) if raw_safety else None

    raw_caption = d.pop("caption_style_json", None)
    d["caption_style"] = json.loads(raw_caption) if raw_caption else None

    raw_platforms = d.pop("auto_publish_platforms_json", None)
    d["auto_publish_platforms"] = json.loads(raw_platforms) if raw_platforms else []

    raw_phrases = d.pop("custom_trigger_phrases_json", None)
    if raw_phrases:
        d["custom_trigger_phrases"] = CustomTriggerPhrases.model_validate(
            json.loads(raw_phrases)
        )
    else:
        d["custom_trigger_phrases"] = CustomTriggerPhrases()

    return BrandKitRow.model_validate(d)


class BrandKitsRepo:
    """CRUD for tenant brand kits.

    `is_default=True` is enforced as at-most-one-per-tenant via a partial
    unique index in migration 006 - `set_default(kit_id)` performs the
    atomic swap by clearing other rows in the same transaction.
    """

    def __init__(self, db: Database):
        self._db = db

    async def create(
        self,
        *,
        name: str,
        primary_color: str,
        accent_color: str,
        text_color: str = "#FFFFFF",
        font_family: str = "Inter",
        font_weight: int = 800,
        default_layout: str = "pip",
        is_default: bool = False,
        logo_url: str | None = None,
        logo_dark_url: str | None = None,
        watermark_url: str | None = None,
        intro_sting_url: str | None = None,
        outro_sting_url: str | None = None,
        caption_style: dict[str, object] | None = None,
        handle_tiktok: str | None = None,
        handle_youtube: str | None = None,
        handle_instagram: str | None = None,
        handle_kick: str | None = None,
        ai_generated: bool = False,
        ai_prompt: str | None = None,
        ai_provider: str | None = None,
        auto_publish_enabled: bool = False,
        auto_publish_platforms: list[str] | None = None,
        auto_publish_delay_min: int = 60,
        custom_trigger_phrases: CustomTriggerPhrases | None = None,
        safe_schedule_enabled: bool = False,
        safety_policy: dict[str, object] | None = None,
        content_timezone: str = "UTC",
        show_chalybclip_outro: bool = True,
    ) -> BrandKitRow:
        tenant_id = current_tenant_id()
        kit_id = new_id("brk")
        now = _now()
        conn = await self._db.connect()
        if is_default:
            await conn.execute(
                "UPDATE brand_kits SET is_default = 0, updated_at = ? "
                "WHERE tenant_id = ? AND is_default = 1",
                (now, tenant_id),
            )
        # _BRAND_KIT_COLS lists 44 columns (40 + 3 safe-trap cols from
        # migration 042 + show_chalybclip_outro from migration 050) — keep
        # the placeholder count in lockstep or sqlite raises "incorrect
        # number of bindings".
        await conn.execute(
            f"INSERT INTO brand_kits ({_BRAND_KIT_COLS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                kit_id, tenant_id, name, 1 if is_default else 0,
                primary_color, accent_color, text_color, font_family, font_weight,
                logo_url, logo_dark_url, watermark_url, intro_sting_url, outro_sting_url,
                json.dumps(caption_style) if caption_style is not None else None,
                default_layout,
                handle_tiktok, handle_youtube, handle_instagram, handle_kick,
                1 if ai_generated else 0, ai_prompt, ai_provider,
                1 if auto_publish_enabled else 0,
                json.dumps(auto_publish_platforms) if auto_publish_platforms else None,
                auto_publish_delay_min,
                json.dumps(
                    (custom_trigger_phrases or CustomTriggerPhrases()).model_dump()
                ),
                # Slice H.1 — user-level editor prefs. New rows start
                # with platform/toggles unset; the dashboard's auto-save
                # endpoint fills them as the operator interacts.
                None,    # default_platform
                0,       # banner_enabled_default
                0,       # banner_show_context_default
                0,       # banner_show_safezones_default
                # Slice I.1 — clip style preset + banner variant + top hook.
                # Defaults to "repost_page_viral" via get_clip_style() when
                # read back; we write NULL here so a brand-new tenant
                # transparently inherits the system default until they pick.
                None,    # clip_style
                None,    # bottom_banner_style
                0,       # banner_live_badge_default
                0,       # top_hook_enabled_default
                None,    # top_hook_style_default
                # Slice K.5 — target platform default. NULL → editor
                # falls back to PLATFORM_PRESETS["all"].
                None,    # target_platform
                # Slice O.1 — show_chalybclip_credit defaults ON.
                1,
                # End-card outro toggle (migration 050) — defaults ON;
                # the create() caller can override for paid tenants.
                1 if show_chalybclip_outro else 0,
                # Safe trap (migration 042) — advisory by default.
                1 if safe_schedule_enabled else 0,
                json.dumps(safety_policy) if safety_policy else None,
                content_timezone,
                now, now,
            ),
        )
        await conn.commit()
        out = await self.get(kit_id)
        assert out is not None
        return out

    async def get(self, kit_id: str) -> BrandKitRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_BRAND_KIT_COLS} FROM brand_kits "
            "WHERE id = ? AND tenant_id = ?",
            (kit_id, tenant_id),
        )
        row = await cur.fetchone()
        return _brand_kit_from_row(row) if row else None

    async def list_for_tenant(self) -> list[BrandKitRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_BRAND_KIT_COLS} FROM brand_kits "
            "WHERE tenant_id = ? ORDER BY is_default DESC, created_at",
            (tenant_id,),
        )
        return [_brand_kit_from_row(r) for r in await cur.fetchall()]

    async def get_default(self) -> BrandKitRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_BRAND_KIT_COLS} FROM brand_kits "
            "WHERE tenant_id = ? AND is_default = 1 LIMIT 1",
            (tenant_id,),
        )
        row = await cur.fetchone()
        return _brand_kit_from_row(row) if row else None

    async def set_default(self, kit_id: str) -> BrandKitRow:
        """Atomically promote kit_id to the tenant default."""
        existing = await self.get(kit_id)
        if existing is None:
            raise ChalybClipError(f"brand kit {kit_id!r} not found")
        tenant_id = current_tenant_id()
        now = _now()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE brand_kits SET is_default = 0, updated_at = ? "
            "WHERE tenant_id = ? AND is_default = 1",
            (now, tenant_id),
        )
        await conn.execute(
            "UPDATE brand_kits SET is_default = 1, updated_at = ? "
            "WHERE id = ? AND tenant_id = ?",
            (now, kit_id, tenant_id),
        )
        await conn.commit()
        out = await self.get(kit_id)
        assert out is not None
        return out

    async def update(
        self,
        kit_id: str,
        *,
        name: str | None = None,
        primary_color: str | None = None,
        accent_color: str | None = None,
        text_color: str | None = None,
        font_family: str | None = None,
        font_weight: int | None = None,
        default_layout: str | None = None,
        caption_style: dict[str, object] | None = None,
        handle_tiktok: str | None = None,
        handle_youtube: str | None = None,
        handle_instagram: str | None = None,
        handle_kick: str | None = None,
        auto_publish_enabled: bool | None = None,
        auto_publish_platforms: list[str] | None = None,
        auto_publish_delay_min: int | None = None,
        custom_trigger_phrases: CustomTriggerPhrases | None = None,
        logo_url: str | None = None,
        ai_generated: bool | None = None,
        ai_prompt: str | None = None,
        ai_provider: str | None = None,
        # Slice H.1 — user-level editor prefs.
        default_platform: str | None = None,
        banner_enabled_default: bool | None = None,
        banner_show_context_default: bool | None = None,
        banner_show_safezones_default: bool | None = None,
        # Slice I.1 — clip style preset + banner variant + top hook.
        clip_style: str | None = None,
        bottom_banner_style: str | None = None,
        banner_live_badge_default: bool | None = None,
        top_hook_enabled_default: bool | None = None,
        top_hook_style_default: str | None = None,
        # Slice K.5 — operator's default target platform.
        target_platform: str | None = None,
        # Safe trap (migration 042).
        safe_schedule_enabled: bool | None = None,
        safety_policy: dict[str, object] | None = None,
        content_timezone: str | None = None,
        # End-card outro toggle (migration 050).
        show_chalybclip_outro: bool | None = None,
    ) -> BrandKitRow:
        """Partial update - only non-None args are applied."""
        existing = await self.get(kit_id)
        if existing is None:
            raise ChalybClipError(f"brand kit {kit_id!r} not found")
        sets: list[str] = []
        values: list[object] = []
        if name is not None:
            sets.append("name = ?")
            values.append(name)
        if primary_color is not None:
            sets.append("primary_color = ?")
            values.append(primary_color)
        if accent_color is not None:
            sets.append("accent_color = ?")
            values.append(accent_color)
        if text_color is not None:
            sets.append("text_color = ?")
            values.append(text_color)
        if font_family is not None:
            sets.append("font_family = ?")
            values.append(font_family)
        if font_weight is not None:
            sets.append("font_weight = ?")
            values.append(font_weight)
        if default_layout is not None:
            sets.append("default_layout = ?")
            values.append(default_layout)
        if caption_style is not None:
            sets.append("caption_style_json = ?")
            values.append(json.dumps(caption_style))
        for col, val in (
            ("handle_tiktok", handle_tiktok),
            ("handle_youtube", handle_youtube),
            ("handle_instagram", handle_instagram),
            ("handle_kick", handle_kick),
        ):
            if val is not None:
                sets.append(f"{col} = ?")
                values.append(val if val else None)
        if auto_publish_enabled is not None:
            sets.append("auto_publish_enabled = ?")
            values.append(1 if auto_publish_enabled else 0)
        if auto_publish_platforms is not None:
            sets.append("auto_publish_platforms_json = ?")
            values.append(json.dumps(auto_publish_platforms))
        if auto_publish_delay_min is not None:
            sets.append("auto_publish_delay_min = ?")
            values.append(auto_publish_delay_min)
        if custom_trigger_phrases is not None:
            sets.append("custom_trigger_phrases_json = ?")
            values.append(json.dumps(custom_trigger_phrases.model_dump()))
        if logo_url is not None:
            sets.append("logo_url = ?")
            values.append(logo_url if logo_url else None)
        if ai_generated is not None:
            sets.append("ai_generated = ?")
            values.append(1 if ai_generated else 0)
        if ai_prompt is not None:
            sets.append("ai_prompt = ?")
            values.append(ai_prompt if ai_prompt else None)
        if ai_provider is not None:
            sets.append("ai_provider = ?")
            values.append(ai_provider if ai_provider else None)
        # Slice H.1 — user-level editor prefs partial-update.
        if default_platform is not None:
            sets.append("default_platform = ?")
            values.append(default_platform if default_platform else None)
        if banner_enabled_default is not None:
            sets.append("banner_enabled_default = ?")
            values.append(1 if banner_enabled_default else 0)
        if banner_show_context_default is not None:
            sets.append("banner_show_context_default = ?")
            values.append(1 if banner_show_context_default else 0)
        if banner_show_safezones_default is not None:
            sets.append("banner_show_safezones_default = ?")
            values.append(1 if banner_show_safezones_default else 0)
        # Slice I.1 — clip style preset + banner variant + top hook.
        if clip_style is not None:
            sets.append("clip_style = ?")
            values.append(clip_style if clip_style else None)
        if bottom_banner_style is not None:
            sets.append("bottom_banner_style = ?")
            values.append(bottom_banner_style if bottom_banner_style else None)
        if banner_live_badge_default is not None:
            sets.append("banner_live_badge_default = ?")
            values.append(1 if banner_live_badge_default else 0)
        if top_hook_enabled_default is not None:
            sets.append("top_hook_enabled_default = ?")
            values.append(1 if top_hook_enabled_default else 0)
        if top_hook_style_default is not None:
            sets.append("top_hook_style_default = ?")
            values.append(top_hook_style_default if top_hook_style_default else None)
        # Slice K.5 — operator's default target platform.
        if target_platform is not None:
            sets.append("target_platform = ?")
            values.append(target_platform if target_platform else None)
        # Safe trap (migration 042).
        if safe_schedule_enabled is not None:
            sets.append("safe_schedule_enabled = ?")
            values.append(1 if safe_schedule_enabled else 0)
        if safety_policy is not None:
            sets.append("safety_policy_json = ?")
            values.append(json.dumps(safety_policy) if safety_policy else None)
        if content_timezone is not None:
            sets.append("content_timezone = ?")
            values.append(content_timezone)
        if show_chalybclip_outro is not None:
            sets.append("show_chalybclip_outro = ?")
            values.append(1 if show_chalybclip_outro else 0)
        if not sets:
            return existing
        sets.append("updated_at = ?")
        values.append(_now())
        tenant_id = current_tenant_id()
        values.extend([kit_id, tenant_id])
        conn = await self._db.connect()
        await conn.execute(
            f"UPDATE brand_kits SET {', '.join(sets)} "
            "WHERE id = ? AND tenant_id = ?",
            tuple(values),
        )
        await conn.commit()
        out = await self.get(kit_id)
        assert out is not None
        return out

    async def delete(self, kit_id: str) -> None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM brand_kits WHERE id = ? AND tenant_id = ?",
            (kit_id, tenant_id),
        )
        await conn.commit()


# ---------- Drive watches (voice-markers spec slice E.4) ----------


_DRIVE_WATCH_COLS = (
    "id, tenant_id, folder_id, folder_name, refresh_token, access_token, "
    "access_token_expires_at, last_polled_at, seen_file_ids_json, enabled, "
    "created_at, updated_at"
)


def _drive_watch_from_row(row: aiosqlite.Row) -> DriveWatchRow:
    d = dict(row)
    seen_blob = d.pop("seen_file_ids_json")
    d["seen_file_ids"] = json.loads(seen_blob) if seen_blob else []
    d["enabled"] = bool(d["enabled"])
    return DriveWatchRow.model_validate(d)


class DriveWatchesRepo:
    """CRUD for `drive_watches`. Polled by `chalybclip drive poll` and the
    in-process scheduler. (voice-markers spec slice E.4)
    """

    def __init__(self, db: Database):
        self._db = db

    async def create(
        self,
        *,
        folder_id: str,
        folder_name: str | None,
        refresh_token: str,
        access_token: str | None = None,
        access_token_expires_at: str | None = None,
    ) -> DriveWatchRow:
        tenant_id = current_tenant_id()
        watch_id = new_id("drv")
        now = _now()
        conn = await self._db.connect()
        await conn.execute(
            f"INSERT INTO drive_watches ({_DRIVE_WATCH_COLS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, '[]', 1, ?, ?)",
            (
                watch_id,
                tenant_id,
                folder_id,
                folder_name,
                refresh_token,
                access_token,
                access_token_expires_at,
                now,
                now,
            ),
        )
        await conn.commit()
        out = await self.get(watch_id)
        assert out is not None
        return out

    async def get(self, watch_id: str) -> DriveWatchRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_DRIVE_WATCH_COLS} FROM drive_watches "
            "WHERE id = ? AND tenant_id = ?",
            (watch_id, tenant_id),
        )
        row = await cur.fetchone()
        return _drive_watch_from_row(row) if row else None

    async def list_for_tenant(self) -> list[DriveWatchRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_DRIVE_WATCH_COLS} FROM drive_watches "
            "WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        )
        return [_drive_watch_from_row(r) for r in await cur.fetchall()]

    async def mark_polled(
        self,
        watch_id: str,
        *,
        seen_file_ids: list[str],
        last_polled_at: str | None,
    ) -> None:
        """Persist progress after a poll pass.

        Tenancy-checked even though the poller has already bound — same
        defense-in-depth pattern other repos use."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE drive_watches SET seen_file_ids_json = ?, "
            "last_polled_at = ?, updated_at = ? "
            "WHERE id = ? AND tenant_id = ?",
            (
                json.dumps(seen_file_ids),
                last_polled_at,
                _now(),
                watch_id,
                tenant_id,
            ),
        )
        await conn.commit()

    async def set_enabled(self, watch_id: str, enabled: bool) -> DriveWatchRow:
        """Pause / resume a watch without deleting (preserves seen_file_ids)."""
        existing = await self.get(watch_id)
        if existing is None:
            raise ChalybClipError(f"drive watch {watch_id!r} not found")
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE drive_watches SET enabled = ?, updated_at = ? "
            "WHERE id = ? AND tenant_id = ?",
            (1 if enabled else 0, _now(), watch_id, tenant_id),
        )
        await conn.commit()
        out = await self.get(watch_id)
        assert out is not None
        return out

    async def delete(self, watch_id: str) -> None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM drive_watches WHERE id = ? AND tenant_id = ?",
            (watch_id, tenant_id),
        )
        await conn.commit()


# ---------- Channel watches (auto-ingest from creator channels) ----------


_CHANNEL_WATCH_COLS = (
    "id, tenant_id, platform, channel_url, channel_label, persona_id, "
    "language, last_polled_at, seen_video_ids_json, max_per_poll, enabled, "
    "created_at, updated_at, polls_per_day, failed_video_attempts_json"
)


def _channel_watch_from_row(row: aiosqlite.Row) -> ChannelWatchRow:
    d = dict(row)
    seen_blob = d.pop("seen_video_ids_json")
    d["seen_video_ids"] = json.loads(seen_blob) if seen_blob else []
    failed_blob = d.pop("failed_video_attempts_json", None)
    d["failed_video_attempts"] = json.loads(failed_blob) if failed_blob else {}
    d["enabled"] = bool(d["enabled"])
    return ChannelWatchRow.model_validate(d)


class ChannelWatchesRepo:
    """CRUD for `channel_watches`. Polled by `chalybclip channel poll` and the
    in-process channel-poll loop to auto-ingest new VODs from a creator's
    YouTube / Twitch / Kick channel. Mirrors `DriveWatchesRepo`."""

    def __init__(self, db: Database):
        self._db = db

    async def create(
        self,
        *,
        platform: str,
        channel_url: str,
        persona_id: str,
        channel_label: str | None = None,
        language: str | None = None,
        max_per_poll: int = 3,
        polls_per_day: int = 1,
    ) -> ChannelWatchRow:
        tenant_id = current_tenant_id()
        watch_id = new_id("chw")
        now = _now()
        conn = await self._db.connect()
        await conn.execute(
            f"INSERT INTO channel_watches ({_CHANNEL_WATCH_COLS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, '[]', ?, 1, ?, ?, ?, '{}')",
            (
                watch_id,
                tenant_id,
                platform,
                channel_url,
                channel_label,
                persona_id,
                language,
                max_per_poll,
                now,
                now,
                max(polls_per_day, 1),
            ),
        )
        await conn.commit()
        out = await self.get(watch_id)
        assert out is not None
        return out

    async def set_polls_per_day(
        self, watch_id: str, polls_per_day: int
    ) -> ChannelWatchRow:
        """Change the poll cadence (times/day). Clamped to >= 1."""
        existing = await self.get(watch_id)
        if existing is None:
            raise ChalybClipError(f"channel watch {watch_id!r} not found")
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE channel_watches SET polls_per_day = ?, updated_at = ? "
            "WHERE id = ? AND tenant_id = ?",
            (max(polls_per_day, 1), _now(), watch_id, tenant_id),
        )
        await conn.commit()
        out = await self.get(watch_id)
        assert out is not None
        return out

    async def get(self, watch_id: str) -> ChannelWatchRow | None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_CHANNEL_WATCH_COLS} FROM channel_watches "
            "WHERE id = ? AND tenant_id = ?",
            (watch_id, tenant_id),
        )
        row = await cur.fetchone()
        return _channel_watch_from_row(row) if row else None

    async def list_for_tenant(self) -> list[ChannelWatchRow]:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_CHANNEL_WATCH_COLS} FROM channel_watches "
            "WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        )
        return [_channel_watch_from_row(r) for r in await cur.fetchall()]

    async def mark_polled(
        self,
        watch_id: str,
        *,
        seen_video_ids: list[str],
        last_polled_at: str | None,
        failed_video_attempts: dict[str, int] | None = None,
    ) -> None:
        """Persist progress after a poll pass (seen-set + per-video failure
        counts + last_polled_at). Omitting `failed_video_attempts` clears the
        retry map — only the poller passes it, and it always passes the full
        recomputed map."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE channel_watches SET seen_video_ids_json = ?, "
            "failed_video_attempts_json = ?, last_polled_at = ?, updated_at = ? "
            "WHERE id = ? AND tenant_id = ?",
            (
                json.dumps(seen_video_ids),
                json.dumps(failed_video_attempts or {}),
                last_polled_at,
                _now(),
                watch_id,
                tenant_id,
            ),
        )
        await conn.commit()

    async def set_enabled(
        self, watch_id: str, enabled: bool
    ) -> ChannelWatchRow:
        """Pause / resume a watch without deleting (preserves seen-set)."""
        existing = await self.get(watch_id)
        if existing is None:
            raise ChalybClipError(f"channel watch {watch_id!r} not found")
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE channel_watches SET enabled = ?, updated_at = ? "
            "WHERE id = ? AND tenant_id = ?",
            (1 if enabled else 0, _now(), watch_id, tenant_id),
        )
        await conn.commit()
        out = await self.get(watch_id)
        assert out is not None
        return out

    async def delete(self, watch_id: str) -> None:
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM channel_watches WHERE id = ? AND tenant_id = ?",
            (watch_id, tenant_id),
        )
        await conn.commit()


_DRIVE_EXPORT_COLS = (
    "tenant_id, enabled, folder_id, folder_name, refresh_token, "
    "access_token, access_token_expires_at, created_at, updated_at"
)


def _drive_export_from_row(row: aiosqlite.Row) -> DriveExportSettingsRow:
    d = dict(row)
    d["enabled"] = bool(d["enabled"])
    return DriveExportSettingsRow.model_validate(d)


class DriveExportSettingsRepo:
    """CRUD for `drive_export_settings` — a tenant's clip → Drive export
    destination (task #31). One row per tenant (tenant_id is the PK).

    The auto-save-on-render hook + the manual per-clip export button
    both read this. Tokens are written by the OAuth connect flow
    (deferred follow-up)."""

    def __init__(self, db: Database):
        self._db = db

    async def get(self) -> DriveExportSettingsRow | None:
        """The current tenant's export settings, or None if they've
        never touched Drive export."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_DRIVE_EXPORT_COLS} FROM drive_export_settings "
            "WHERE tenant_id = ?",
            (tenant_id,),
        )
        row = await cur.fetchone()
        return _drive_export_from_row(row) if row else None

    async def _ensure_row(self) -> None:
        """Insert an empty (disabled, unconnected) row if none exists, so
        the set_* methods can UPDATE unconditionally."""
        tenant_id = current_tenant_id()
        now = _now()
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO drive_export_settings "
            "(tenant_id, enabled, created_at, updated_at) "
            "VALUES (?, 0, ?, ?) "
            "ON CONFLICT DO NOTHING",
            (tenant_id, now, now),
        )
        await conn.commit()

    async def set_enabled(self, enabled: bool) -> DriveExportSettingsRow:
        """Flip the auto-save-on-render toggle."""
        await self._ensure_row()
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE drive_export_settings SET enabled = ?, updated_at = ? "
            "WHERE tenant_id = ?",
            (1 if enabled else 0, _now(), tenant_id),
        )
        await conn.commit()
        out = await self.get()
        assert out is not None
        return out

    async def set_destination(
        self, *, folder_id: str, folder_name: str | None
    ) -> DriveExportSettingsRow:
        """Pick the destination folder (from the connect flow's folder
        picker)."""
        await self._ensure_row()
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE drive_export_settings SET folder_id = ?, "
            "folder_name = ?, updated_at = ? WHERE tenant_id = ?",
            (folder_id, folder_name, _now(), tenant_id),
        )
        await conn.commit()
        out = await self.get()
        assert out is not None
        return out

    async def set_tokens(
        self,
        *,
        refresh_token: str,
        access_token: str | None = None,
        access_token_expires_at: str | None = None,
    ) -> DriveExportSettingsRow:
        """Persist OAuth tokens from the connect flow / a refresh."""
        await self._ensure_row()
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE drive_export_settings SET refresh_token = ?, "
            "access_token = ?, access_token_expires_at = ?, updated_at = ? "
            "WHERE tenant_id = ?",
            (
                refresh_token,
                access_token,
                access_token_expires_at,
                _now(),
                tenant_id,
            ),
        )
        await conn.commit()
        out = await self.get()
        assert out is not None
        return out

    async def disconnect(self) -> None:
        """Clear tokens + destination (operator unlinks their Drive).
        Keeps the row so the enabled toggle's history isn't lost, but
        wipes credentials so exports stop."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE drive_export_settings SET refresh_token = NULL, "
            "access_token = NULL, access_token_expires_at = NULL, "
            "folder_id = NULL, folder_name = NULL, enabled = 0, "
            "updated_at = ? WHERE tenant_id = ?",
            (_now(), tenant_id),
        )
        await conn.commit()



class ZernioPublishesRepo:
    """Local record of every Zernio publish (migration 030).

    Zernio's GET /posts is scoped to the company API key — NOT per
    tenant — so per-tenant publish history must come from this table.
    One row per POST /posts we fired; the dashboard joins live status
    from Zernio by post_id.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(
        self,
        *,
        post_id: str,
        tenant_id: str,
        clip_id: str,
        platforms: list[str],
        content: str | None,
        status: str | None = None,
        options_json: str | None = None,
        scheduled_for: str | None = None,
    ) -> None:
        """Persist one fired publish. Idempotent on post_id — the
        duplicate-resolved path (Zernio 409 → existing post) records
        the same post id again and must not error.

        `status` seeds the row state at create time (drafts record as
        'draft' so the Borradores panel sees them before any webhook
        lands); `options_json` snapshots the per-platform extras for
        the draft re-publish path (migration 033)."""
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO zernio_publishes "
            "(post_id, tenant_id, clip_id, platforms, content, created_at, "
            "status, options_json, scheduled_for) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            (
                post_id,
                tenant_id,
                clip_id,
                ",".join(platforms),
                content or None,
                _now(),
                status,
                options_json,
                scheduled_for,
            ),
        )
        await conn.commit()

    async def list_for_tenant(
        self, limit: int = 25, *, status: str | None = None
    ) -> list[ZernioPublishRow]:
        """Newest-first publish history for the bound tenant; `status`
        narrows to one state (e.g. 'draft' for the Borradores panel)."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        where = "tenant_id = ?"
        params: list[object] = [tenant_id]
        if status is not None:
            where += " AND status = ?"
            params.append(status)
        cur = await conn.execute(
            "SELECT post_id, tenant_id, clip_id, platforms, content, created_at, "
            "status, scheduled_for, platforms_json, updated_at, options_json "
            f"FROM zernio_publishes WHERE {where} "  # fixed fragments, params bound
            "ORDER BY created_at DESC LIMIT ?",
            (*params, int(limit)),
        )
        return [
            ZernioPublishRow.model_validate(dict(r)) for r in await cur.fetchall()
        ]

    async def list_for_calendar(
        self,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 2000,
    ) -> list[ZernioPublishRow]:
        """Calendar-window rows for the bound tenant, bounded IN SQL.

        The calendar buckets a post on the day it goes (or went) out —
        `scheduled_for` when set, else `created_at` — so the window filter
        matches that same day expression. Drafts and deleted tombstones
        never reach the calendar. The window replaces a newest-N slice:
        `list_for_tenant(limit=200)` + in-Python date filtering silently
        dropped older-created rows, so posts scheduled by earlier runs
        vanished from the month view while still holding their slots.
        `limit` is a memory backstop, not pagination. Day bounds are
        ISO-8601 (YYYY-MM-DD), inclusive; substr() keeps the bucketing
        portable across SQLite and Postgres."""
        tenant_id = current_tenant_id()
        where = (
            "tenant_id = ? "
            "AND COALESCE(status, '') NOT IN ('draft', 'deleted')"
        )
        params: list[object] = [tenant_id]
        if date_from:
            where += " AND substr(COALESCE(scheduled_for, created_at), 1, 10) >= ?"
            params.append(date_from)
        if date_to:
            where += " AND substr(COALESCE(scheduled_for, created_at), 1, 10) <= ?"
            params.append(date_to)
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT post_id, tenant_id, clip_id, platforms, content, created_at, "
            "status, scheduled_for, platforms_json, updated_at, options_json "
            f"FROM zernio_publishes WHERE {where} "  # fixed fragments, params bound
            "ORDER BY COALESCE(scheduled_for, created_at) LIMIT ?",
            (*params, int(limit)),
        )
        return [
            ZernioPublishRow.model_validate(dict(r)) for r in await cur.fetchall()
        ]

    async def count_status_groups(self) -> dict[str, int]:
        """Exact per-status row counts for the bound tenant → {status: n}
        (rows with no status yet land under ''). The stats strip needs REAL
        totals — counting over the newest-100 display slice pinned
        "Programados" at exactly the cap for any bigger backlog.

        Synthetic `manual_<clip_id>` rows (mark-published) count as
        'published' regardless of their stored status: they're recorded with
        no status and no webhook ever updates them, which would otherwise
        park every manual mark in the '' (queued) bucket forever. The LIKE
        underscore is escaped — it's a single-char wildcard otherwise."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT CASE WHEN post_id LIKE 'manual\\_%' ESCAPE '\\' "
            "THEN 'published' ELSE COALESCE(status, '') END AS s, "
            "COUNT(*) AS n "
            "FROM zernio_publishes WHERE tenant_id = ? GROUP BY s",
            (tenant_id,),
        )
        return {
            str(dict(r)["s"]): int(dict(r)["n"]) for r in await cur.fetchall()
        }

    async def recorded_clip_ids(self) -> set[str]:
        """Every clip id with ANY publish row for the bound tenant — the
        membership set deciding which published clips get a synthetic
        "manual" history entry. Uncapped: deriving it from the capped
        display slice re-listed clips whose rows had aged past the cap."""
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT DISTINCT clip_id FROM zernio_publishes WHERE tenant_id = ?",
            (tenant_id,),
        )
        return {str(dict(r)["clip_id"]) for r in await cur.fetchall()}

    async def pending_for_stream(self, stream_id: str) -> list[dict[str, str]]:
        """Publish rows still WAITING to fire whose clip belongs to
        `stream_id` — the guard set for stream deletion. Deleting a stream
        cascades its clip rows, and a scheduled Zernio post whose clip row
        is gone 404s at publish time (the vendor fetches our signed media
        URL only when the slot arrives). Pending = the pre-flight states;
        posted/failed/cancelled/draft rows don't block anything.
        """
        tenant_id = current_tenant_id()
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT zp.post_id, zp.clip_id, zp.status, zp.scheduled_for "
            "FROM zernio_publishes zp JOIN clips c ON c.id = zp.clip_id "
            "WHERE zp.tenant_id = ? AND c.tenant_id = ? AND c.stream_id = ? "
            "AND COALESCE(zp.status, '') IN "
            "('scheduled', 'pending', 'queued', 'publishing', 'processing')",
            (tenant_id, tenant_id, stream_id),
        )
        return [
            {
                "post_id": str(d["post_id"]),
                "clip_id": str(d["clip_id"]),
                "status": str(d["status"]),
                "scheduled_for": str(d["scheduled_for"] or ""),
            }
            for d in (dict(r) for r in await cur.fetchall())
        ]

    async def exists_for_clip(self, tenant_id: str, clip_id: str) -> bool:
        """True if this clip already has a LIVE publish record — the
        idempotency guard for hands-free auto-publish (don't re-post on
        pipeline re-runs). Cancelled/deleted rows don't count: an operator
        who cancels a scheduled post is putting the clip back in play, and
        a tombstone that still blocked scheduling black-holed the clip
        permanently (no UI path could ever re-schedule it)."""
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT 1 FROM zernio_publishes "
            "WHERE tenant_id = ? AND clip_id = ? "
            "AND COALESCE(status, '') NOT IN ('cancelled', 'deleted') LIMIT 1",
            (tenant_id, clip_id),
        )
        return await cur.fetchone() is not None

    async def count_by_platform_today(self, tenant_id: str) -> dict[str, int]:
        """Posts already scheduled/published for THIS UTC day, per platform.

        The per-platform daily-cap accounting for the rulebook scheduler: a
        post counts toward the day of its `scheduled_for` (or `created_at` when
        it fired now), excluding failed/deleted/cancelled rows (a cancelled
        post frees its slot — that's the point of cancelling to make room).
        With the per-platform posting model each row is one platform, but we
        split the csv anyway so older multi-platform rows still count.
        Platform ids are canonicalized (twitter, not "x") to match the
        rulebook keys."""
        from chalybclip.publish.pacing import canonical_platform

        day = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d")
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT platforms, scheduled_for, created_at FROM zernio_publishes "
            "WHERE tenant_id = ? "
            "AND COALESCE(status, '') NOT IN ('failed', 'deleted', 'cancelled') "
            "AND substr(COALESCE(scheduled_for, created_at), 1, 10) = ?",
            (tenant_id, day),
        )
        out: dict[str, int] = {}
        for row in await cur.fetchall():
            for p in str(dict(row).get("platforms") or "").split(","):
                key = canonical_platform(p)
                if key:
                    out[key] = out.get(key, 0) + 1
        return out

    async def count_by_platform_by_day(
        self, tenant_id: str, *, from_day: str
    ) -> dict[str, dict[str, int]]:
        """Live posts per platform per UTC day, from `from_day` (YYYY-MM-DD)
        onward → {platform: {day: count}}.

        The cross-sweep counterpart of `count_by_platform_today`: the rulebook
        scheduler rolls overflow onto FUTURE days, and a later planning run
        (same day, new VOD) can't see those rows through the today-only count —
        so each run stacked another `max_per_day` onto every future day. Same
        accounting rules as today's count: a post lands on the day of its
        `scheduled_for` (or `created_at` when it fired now), failed/deleted/
        cancelled rows free their slot, csv platforms are split and
        canonicalized. substr() keeps the day-bucketing portable across SQLite
        and Postgres (ISO-8601 strings sort/prefix identically on both)."""
        from chalybclip.publish.pacing import canonical_platform

        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT platforms, scheduled_for, created_at FROM zernio_publishes "
            "WHERE tenant_id = ? "
            "AND COALESCE(status, '') NOT IN ('failed', 'deleted', 'cancelled') "
            "AND substr(COALESCE(scheduled_for, created_at), 1, 10) >= ?",
            (tenant_id, from_day),
        )
        out: dict[str, dict[str, int]] = {}
        for row in await cur.fetchall():
            d = dict(row)
            day = str(d.get("scheduled_for") or d.get("created_at") or "")[:10]
            if not day:
                continue
            for p in str(d.get("platforms") or "").split(","):
                key = canonical_platform(p)
                if key:
                    per_day = out.setdefault(key, {})
                    per_day[day] = per_day.get(day, 0) + 1
        return out

    async def get_by_post_id(self, post_id: str) -> ZernioPublishRow | None:
        """Tenant-FREE lookup by Zernio post id.

        Webhook deliveries are server-to-server (no bound tenant); the
        post id is the join key that RESOLVES the tenant. Same
        invocation pattern as the tenant-free stream helpers above —
        callers must treat the returned row's tenant_id as the scope.
        """
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT post_id, tenant_id, clip_id, platforms, content, created_at, "
            "status, platforms_json, updated_at, options_json "
            "FROM zernio_publishes WHERE post_id = ?",
            (post_id,),
        )
        row = await cur.fetchone()
        return ZernioPublishRow.model_validate(dict(row)) if row else None

    async def set_status(
        self,
        post_id: str,
        *,
        status: str,
        platforms_json: str | None = None,
    ) -> None:
        """Tenant-FREE status update fed by post.* webhooks.

        Keyed by the Zernio post id (PRIMARY KEY); a webhook for a post
        we never recorded is a silent no-op (e.g. fired from Zernio's
        dashboard directly, outside the hub)."""
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE zernio_publishes SET status = ?, "
            "platforms_json = COALESCE(?, platforms_json), updated_at = ? "
            "WHERE post_id = ?",
            (status, platforms_json, _now(), post_id),
        )
        await conn.commit()

    async def delete(self, post_id: str) -> None:
        """Hard-delete a publish record by post id (tenant-free; post_id is the
        PK). Used by queue reprocessing to scrub a cancelled post's row from
        history entirely. (Since `exists_for_clip` learned to ignore
        cancelled/deleted rows, a tombstone would unblock re-scheduling too —
        reprocess deletes anyway to keep the reprocessed queue's history
        clean.)"""
        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM zernio_publishes WHERE post_id = ?", (post_id,)
        )
        await conn.commit()


class ZernioWhatsappNumbersRepo:
    """WhatsApp number provisioning status (migration 040, phase 12).

    Fed by whatsapp.number.* webhooks; latest status per account wins.
    Tenant-free at rest (keyed by account_id), resolved at read time."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def upsert(
        self, *, account_id: str, status: str, detail: str | None = None
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO zernio_whatsapp_numbers "
            "(account_id, status, detail, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(account_id) DO UPDATE SET "
            "status = excluded.status, detail = excluded.detail, "
            "updated_at = excluded.updated_at",
            (account_id, status, detail, _now()),
        )
        await conn.commit()

    async def list_for_accounts(
        self, account_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not account_ids:
            return []
        placeholders = ",".join("?" for _ in account_ids)
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT account_id, status, detail, updated_at "
            f"FROM zernio_whatsapp_numbers WHERE account_id IN ({placeholders})",
            tuple(account_ids),
        )
        return [dict(r) for r in await cur.fetchall()]


class AutopublishSettingsRepo:
    """Per-tenant auto-publish ("Piloto automático") settings (migration 044).

    When `enabled`, the Publish Center auto-enqueues clips to Zernio with
    their burned-in render. `mode` is on_approve | hands_free; `post_mode`
    is the Zernio publish mode (queue | now); `targets` is a csv of Zernio
    platform ids; `daily_cap` is an anti-spam ceiling per UTC day (0 = off).
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get(self, tenant_id: str) -> dict[str, Any] | None:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT tenant_id, enabled, mode, targets, post_mode, daily_cap, "
            "score_threshold, tag_suffix, "
            "growth_engine, growth_min_score, daily_clip_budget, updated_at "
            "FROM autopublish_settings "
            "WHERE tenant_id = ?",
            (tenant_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["enabled"] = bool(d.get("enabled"))
        d["growth_engine"] = bool(d.get("growth_engine"))
        return d

    async def upsert(
        self,
        tenant_id: str,
        *,
        enabled: bool,
        mode: str,
        targets: str | None,
        post_mode: str,
        daily_cap: int,
        score_threshold: float = 0.6,
        tag_suffix: str = "",
        growth_engine: bool = False,
        growth_min_score: int = 40,
        daily_clip_budget: int | None = None,
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO autopublish_settings "
            "(tenant_id, enabled, mode, targets, post_mode, daily_cap, "
            "score_threshold, tag_suffix, "
            "growth_engine, growth_min_score, daily_clip_budget, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(tenant_id) DO UPDATE SET "
            "enabled = excluded.enabled, mode = excluded.mode, "
            "targets = excluded.targets, post_mode = excluded.post_mode, "
            "daily_cap = excluded.daily_cap, "
            "score_threshold = excluded.score_threshold, "
            "tag_suffix = excluded.tag_suffix, "
            "growth_engine = excluded.growth_engine, "
            "growth_min_score = excluded.growth_min_score, "
            "daily_clip_budget = excluded.daily_clip_budget, "
            "updated_at = excluded.updated_at",
            (
                tenant_id, 1 if enabled else 0, mode, targets, post_mode,
                daily_cap, score_threshold, tag_suffix,
                1 if growth_engine else 0, growth_min_score, daily_clip_budget,
                _now(),
            ),
        )
        await conn.commit()


class PlatformPacingRulesRepo:
    """Per-tenant per-platform publishing rulebook overrides (migration 052).

    Rows are OVERRIDES: `effective_rules` overlays them onto the shipped
    `DEFAULT_PLATFORM_RULES`, so an untuned tenant gets the conservative
    defaults and a tuned platform gets exactly its saved row. Platform ids are
    stored canonical (twitter, not "x")."""

    _COLS = (
        "platform", "max_per_day", "min_gap_minutes", "caption_min_chars",
        "caption_max_chars", "hashtag_min", "hashtag_max", "jitter_minutes",
        "caption_style", "enabled",
    )

    def __init__(self, db: Database) -> None:
        self._db = db

    async def overrides(self, tenant_id: str) -> dict[str, PlatformRule]:
        """The tenant's saved rule rows as PlatformRule objects, keyed by
        canonical platform id. Empty dict when nothing is tuned."""
        from chalybclip.publish.pacing import PlatformRule, canonical_platform

        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT " + ", ".join(self._COLS) + " FROM platform_pacing_rules "
            "WHERE tenant_id = ?",
            (tenant_id,),
        )
        rows = await cur.fetchall()
        out: dict[str, PlatformRule] = {}
        for row in rows:
            d = dict(row)
            d["enabled"] = bool(d.get("enabled"))
            key = canonical_platform(str(d["platform"]))
            out[key] = PlatformRule(**{**d, "platform": key})
        return out

    async def effective_rules(self, tenant_id: str) -> dict[str, PlatformRule]:
        """Defaults overlaid with this tenant's overrides — the rulebook the
        scheduler/allocator should actually use."""
        from chalybclip.publish.pacing import DEFAULT_PLATFORM_RULES

        rules = dict(DEFAULT_PLATFORM_RULES)
        rules.update(await self.overrides(tenant_id))
        return rules

    async def upsert(self, tenant_id: str, rule: PlatformRule) -> None:
        from chalybclip.publish.pacing import canonical_platform

        key = canonical_platform(rule.platform)
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO platform_pacing_rules "
            "(tenant_id, platform, max_per_day, min_gap_minutes, "
            "caption_min_chars, caption_max_chars, hashtag_min, hashtag_max, "
            "jitter_minutes, caption_style, enabled, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(tenant_id, platform) DO UPDATE SET "
            "max_per_day = excluded.max_per_day, "
            "min_gap_minutes = excluded.min_gap_minutes, "
            "caption_min_chars = excluded.caption_min_chars, "
            "caption_max_chars = excluded.caption_max_chars, "
            "hashtag_min = excluded.hashtag_min, "
            "hashtag_max = excluded.hashtag_max, "
            "jitter_minutes = excluded.jitter_minutes, "
            "caption_style = excluded.caption_style, "
            "enabled = excluded.enabled, updated_at = excluded.updated_at",
            (
                tenant_id, key, rule.max_per_day, rule.min_gap_minutes,
                rule.caption_min_chars, rule.caption_max_chars, rule.hashtag_min,
                rule.hashtag_max, rule.jitter_minutes, rule.caption_style,
                1 if rule.enabled else 0, _now(),
            ),
        )
        await conn.commit()

    async def delete(self, tenant_id: str, platform: str) -> None:
        """Drop a platform's override → it reverts to the shipped default."""
        from chalybclip.publish.pacing import canonical_platform

        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM platform_pacing_rules WHERE tenant_id = ? AND platform = ?",
            (tenant_id, canonical_platform(platform)),
        )
        await conn.commit()


class PlatformCooldownsRepo:
    """Per-tenant per-platform publish cooldowns (migration 057) — the
    abuse/rate-limit backoff. A platform with a future `until` is parked: the
    scheduler skips it so we stop feeding a platform that's flagging us."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def set_cooldown(
        self, tenant_id: str, platform: str, *, until: str, reason: str | None = None
    ) -> None:
        from chalybclip.publish.pacing import canonical_platform

        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO platform_cooldowns (tenant_id, platform, until, reason, "
            "updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(tenant_id, platform) DO UPDATE SET "
            "until = excluded.until, reason = excluded.reason, "
            "updated_at = excluded.updated_at",
            (tenant_id, canonical_platform(platform), until, reason, _now()),
        )
        await conn.commit()

    async def active(self, tenant_id: str) -> dict[str, str]:
        """{platform: until_iso} for platforms still cooling down (until > now).
        Platforms whose cooldown has lapsed are omitted (and pruned)."""
        now = _dt.datetime.now(_dt.UTC).isoformat()
        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM platform_cooldowns WHERE tenant_id = ? AND until <= ?",
            (tenant_id, now),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT platform, until FROM platform_cooldowns WHERE tenant_id = ?",
            (tenant_id,),
        )
        return {str(r[0]): str(r[1]) for r in await cur.fetchall()}

    async def clear_stale_by_reason(
        self, tenant_id: str, *, reason: str, keep: set[str]
    ) -> int:
        """Delete this tenant's `reason` rows whose platform is NOT in `keep`
        (canonical ids). Returns rows deleted.

        The self-heal for evidence-derived cooldowns: the refresh recomputes
        which platforms the CURRENT failed-post evidence still justifies, and
        anything this reason wrote earlier that the evidence no longer
        supports (e.g. rows the pre-fix code re-armed from `now` forever) is
        dropped instead of parking the platform until its bogus `until`
        passes. Rows written with a different reason are never touched."""
        from chalybclip.publish.pacing import canonical_platform

        keep_canon = {canonical_platform(p) for p in keep if p}
        conn = await self._db.connect()
        params: list[object] = [tenant_id, reason]
        sql = (
            "DELETE FROM platform_cooldowns WHERE tenant_id = ? AND reason = ?"
        )
        if keep_canon:
            placeholders = ", ".join("?" for _ in keep_canon)
            sql += f" AND platform NOT IN ({placeholders})"  # bound below
            params.extend(sorted(keep_canon))
        cur = await conn.execute(sql, tuple(params))
        await conn.commit()
        return int(cur.rowcount if cur.rowcount is not None else 0)


class GrowthScoresRepo:
    """Pre-publish Growth Score cache, one row per clip (migration 053).

    Stores the whole `GrowthScoreCard` as JSON plus the two hot columns the
    allocator + fatigue passes query (overall_score, content_tags). Idempotent
    per clip: `record` overwrites."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(
        self,
        *,
        tenant_id: str,
        clip_id: str,
        overall_score: int,
        decision: str,
        content_tags: list[str],
        card_json: str,
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO clip_growth_scores "
            "(clip_id, tenant_id, overall_score, decision, content_tags, "
            "card_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(clip_id) DO UPDATE SET "
            "overall_score = excluded.overall_score, "
            "decision = excluded.decision, content_tags = excluded.content_tags, "
            "card_json = excluded.card_json, created_at = excluded.created_at",
            (
                clip_id, tenant_id, overall_score, decision,
                json.dumps(content_tags), card_json, _now(),
            ),
        )
        await conn.commit()

    async def get(self, tenant_id: str, clip_id: str) -> dict[str, Any] | None:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT clip_id, tenant_id, overall_score, decision, content_tags, "
            "card_json, created_at FROM clip_growth_scores "
            "WHERE tenant_id = ? AND clip_id = ?",
            (tenant_id, clip_id),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["content_tags"] = json.loads(d.get("content_tags") or "[]")
        return d

    async def recent_content_tags(
        self, tenant_id: str, *, limit: int = 12
    ) -> list[list[str]]:
        """Content tags of the most recently scored clips — the fatigue history
        that spaces similar themes. Newest first."""
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT content_tags FROM clip_growth_scores WHERE tenant_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (tenant_id, int(limit)),
        )
        rows = await cur.fetchall()
        out: list[list[str]] = []
        for row in rows:
            try:
                tags = json.loads(dict(row).get("content_tags") or "[]")
            except (ValueError, TypeError):
                tags = []
            if isinstance(tags, list):
                out.append([str(t) for t in tags])
        return out


class AutoprogLocksRepo:
    """Cross-worker lock for the bulk auto-program run (migration 055).

    `acquire` is atomic and portable: clear any stale lock, then INSERT … ON
    CONFLICT DO NOTHING and read back the token. We won iff the stored token is
    ours. `refresh` heartbeats a long run so it isn't reclaimed mid-flight;
    `release` drops the lock when the run ends."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def acquire(self, tenant_id: str, *, stale_seconds: int = 1200) -> str | None:
        """Try to take the lock. Returns a token on success, None if a fresh
        lock is already held by another run."""
        import secrets

        token = secrets.token_hex(12)
        cutoff = (
            _dt.datetime.now(_dt.UTC) - _dt.timedelta(seconds=stale_seconds)
        ).isoformat()
        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM autoprog_locks WHERE claimed_at < ?", (cutoff,)
        )
        await conn.execute(
            "INSERT INTO autoprog_locks (tenant_id, token, claimed_at) "
            "VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
            (tenant_id, token, _now()),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT token FROM autoprog_locks WHERE tenant_id = ?", (tenant_id,)
        )
        row = await cur.fetchone()
        return token if row is not None and row[0] == token else None

    async def refresh(self, tenant_id: str, token: str) -> None:
        """Heartbeat — bump claimed_at so a long run keeps its lock."""
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE autoprog_locks SET claimed_at = ? "
            "WHERE tenant_id = ? AND token = ?",
            (_now(), tenant_id, token),
        )
        await conn.commit()

    async def release(self, tenant_id: str, token: str) -> None:
        """Drop the lock (only if we still hold it)."""
        conn = await self._db.connect()
        await conn.execute(
            "DELETE FROM autoprog_locks WHERE tenant_id = ? AND token = ?",
            (tenant_id, token),
        )
        await conn.commit()


class ZernioCommunityRepo:
    """Community-notification settings + notify ledger (migration 039).

    Settings are per-tenant. The ledger is tenant-free at the lookup
    layer (the webhook processor resolves the tenant) but every row
    carries tenant_id. `claim_notification` is the once-only + loop
    guard for announce-on-publish."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def get_settings(self, tenant_id: str) -> dict[str, Any] | None:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT tenant_id, enabled, discord_account_id, telegram_account_id, "
            "brand_name, brand_avatar_url, weekly_digest, updated_at "
            "FROM zernio_community_settings WHERE tenant_id = ?",
            (tenant_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["enabled"] = bool(d.get("enabled"))
        d["weekly_digest"] = bool(d.get("weekly_digest"))
        return d

    async def upsert_settings(
        self,
        tenant_id: str,
        *,
        enabled: bool,
        discord_account_id: str | None,
        telegram_account_id: str | None,
        brand_name: str | None,
        brand_avatar_url: str | None,
        weekly_digest: bool,
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO zernio_community_settings "
            "(tenant_id, enabled, discord_account_id, telegram_account_id, "
            "brand_name, brand_avatar_url, weekly_digest, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(tenant_id) DO UPDATE SET "
            "enabled = excluded.enabled, "
            "discord_account_id = excluded.discord_account_id, "
            "telegram_account_id = excluded.telegram_account_id, "
            "brand_name = excluded.brand_name, "
            "brand_avatar_url = excluded.brand_avatar_url, "
            "weekly_digest = excluded.weekly_digest, "
            "updated_at = excluded.updated_at",
            (
                tenant_id, 1 if enabled else 0, discord_account_id,
                telegram_account_id, brand_name, brand_avatar_url,
                1 if weekly_digest else 0, _now(),
            ),
        )
        await conn.commit()

    async def list_digest_tenants(self) -> list[dict[str, Any]]:
        """Settings rows with the weekly digest enabled (for the cron)."""
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT tenant_id, discord_account_id, telegram_account_id "
            "FROM zernio_community_settings WHERE weekly_digest = 1 AND enabled = 1",
        )
        return [dict(r) for r in await cur.fetchall()]

    async def claim_notification(
        self, *, source_post_id: str, tenant_id: str
    ) -> bool:
        """Claim the single announcement for `source_post_id`. False if
        already claimed (at-least-once redelivery)."""
        conn = await self._db.connect()
        cur = await conn.execute(
            "INSERT INTO zernio_community_notifications "
            "(source_post_id, tenant_id, sent_at) VALUES (?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            (source_post_id, tenant_id, _now()),
        )
        await conn.commit()
        return cur.rowcount > 0

    async def set_notification_post(
        self, *, source_post_id: str, notification_post_id: str
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE zernio_community_notifications "
            "SET notification_post_id = ? WHERE source_post_id = ?",
            (notification_post_id, source_post_id),
        )
        await conn.commit()

    async def is_notification_post(self, post_id: str) -> bool:
        """True if `post_id` is an announcement WE created — the loop
        guard so a notification's own post.published doesn't re-announce."""
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT 1 FROM zernio_community_notifications "
            "WHERE notification_post_id = ? LIMIT 1",
            (post_id,),
        )
        return (await cur.fetchone()) is not None


class ZernioBroadcastLogRepo:
    """Per-tenant daily broadcast-send log (migration 038) — the
    anti-spam guardrail. Tenant_id is explicit (the route passes it).
    A row is written only when a broadcast actually SENDS."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def count_for_day(self, tenant_id: str, *, day: str) -> int:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT COUNT(*) AS n FROM zernio_broadcast_log "
            "WHERE tenant_id = ? AND day = ?",
            (tenant_id, day),
        )
        row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def record(
        self, tenant_id: str, *, broadcast_id: str, day: str
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO zernio_broadcast_log "
            "(id, tenant_id, broadcast_id, day, sent_at) VALUES (?, ?, ?, ?, ?)",
            (new_id("bcl"), tenant_id, broadcast_id, day, _now()),
        )
        await conn.commit()


class ZernioInboxRepo:
    """Comments + DM conversations/messages + contacts (migration 037).

    Tenant-free at rest — keyed by account_id (inbox webhooks carry no
    profileId), resolved to a tenant at read time by matching against
    the tenant's connected accounts. Webhook-first: the event processor
    writes here, REST is backfill, the UI reads local state."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ---- comments ----

    async def upsert_comment(
        self,
        *,
        account_id: str,
        comment_id: str,
        post_id: str | None,
        platform_post_id: str | None,
        platform: str | None,
        text: str | None,
        author_id: str | None,
        author_name: str | None,
        author_username: str | None,
        is_reply: bool,
        parent_id: str | None,
        created_at: str | None,
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO zernio_comments "
            "(account_id, comment_id, post_id, platform_post_id, platform, text, "
            "author_id, author_name, author_username, is_reply, parent_id, "
            "status, created_at, received_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?) "
            "ON CONFLICT(account_id, comment_id) DO UPDATE SET "
            "text = excluded.text, status = "
            "CASE WHEN zernio_comments.status = 'hidden' THEN 'hidden' "
            "ELSE 'active' END",
            (
                account_id, comment_id, post_id, platform_post_id, platform, text,
                author_id, author_name, author_username, 1 if is_reply else 0,
                parent_id, created_at, _now(),
            ),
        )
        await conn.commit()

    async def set_comment_status(
        self, *, account_id: str, comment_id: str, status: str
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE zernio_comments SET status = ? "
            "WHERE account_id = ? AND comment_id = ?",
            (status, account_id, comment_id),
        )
        await conn.commit()

    async def list_comments(
        self,
        account_ids: list[str],
        *,
        platform_post_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not account_ids:
            return []
        placeholders = ",".join("?" for _ in account_ids)
        where = f"account_id IN ({placeholders})"
        params: list[object] = list(account_ids)
        if platform_post_id:
            where += " AND platform_post_id = ?"
            params.append(platform_post_id)
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT account_id, comment_id, post_id, platform_post_id, platform, "
            "text, author_id, author_name, author_username, is_reply, parent_id, "
            "status, created_at FROM zernio_comments "
            f"WHERE {where} ORDER BY created_at DESC LIMIT ?",  # fixed frags, bound
            (*params, int(limit)),
        )
        return [dict(r) for r in await cur.fetchall()]

    # ---- conversations + messages ----

    async def upsert_conversation(
        self,
        *,
        account_id: str,
        conversation_id: str,
        platform: str | None,
        participant_id: str | None,
        participant_name: str | None,
        participant_username: str | None,
        status: str | None,
        last_message_at: str | None,
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO zernio_conversations "
            "(account_id, conversation_id, platform, participant_id, "
            "participant_name, participant_username, status, last_message_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(account_id, conversation_id) DO UPDATE SET "
            "participant_name = COALESCE(excluded.participant_name, "
            "  zernio_conversations.participant_name), "
            "participant_username = COALESCE(excluded.participant_username, "
            "  zernio_conversations.participant_username), "
            "status = COALESCE(excluded.status, zernio_conversations.status), "
            "last_message_at = COALESCE(excluded.last_message_at, "
            "  zernio_conversations.last_message_at), "
            "updated_at = excluded.updated_at",
            (
                account_id, conversation_id, platform, participant_id,
                participant_name, participant_username, status or "active",
                last_message_at, _now(),
            ),
        )
        await conn.commit()

    async def set_conversation_status(
        self, *, account_id: str, conversation_id: str, status: str
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE zernio_conversations SET status = ?, updated_at = ? "
            "WHERE account_id = ? AND conversation_id = ?",
            (status, _now(), account_id, conversation_id),
        )
        await conn.commit()

    async def list_conversations(
        self, account_ids: list[str], *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if not account_ids:
            return []
        placeholders = ",".join("?" for _ in account_ids)
        where = f"account_id IN ({placeholders})"
        params: list[object] = list(account_ids)
        if status:
            where += " AND status = ?"
            params.append(status)
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT account_id, conversation_id, platform, participant_id, "
            "participant_name, participant_username, status, last_message_at "
            f"FROM zernio_conversations WHERE {where} "  # fixed frags, bound
            "ORDER BY last_message_at DESC LIMIT ?",
            (*params, int(limit)),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def upsert_message(
        self,
        *,
        account_id: str,
        message_id: str,
        conversation_id: str | None,
        platform: str | None,
        direction: str | None,
        text: str | None,
        sent_at: str | None,
        is_read: bool,
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO zernio_messages "
            "(account_id, message_id, conversation_id, platform, direction, text, "
            "sent_at, is_read, received_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(account_id, message_id) DO UPDATE SET "
            "text = excluded.text, is_read = excluded.is_read",
            (
                account_id, message_id, conversation_id, platform, direction, text,
                sent_at, 1 if is_read else 0, _now(),
            ),
        )
        await conn.commit()

    async def list_messages(
        self, account_ids: list[str], *, conversation_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        if not account_ids:
            return []
        placeholders = ",".join("?" for _ in account_ids)
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT account_id, message_id, conversation_id, platform, direction, "
            "text, sent_at, is_read FROM zernio_messages "
            f"WHERE account_id IN ({placeholders}) AND conversation_id = ? "
            "ORDER BY sent_at ASC LIMIT ?",
            (*account_ids, conversation_id, int(limit)),
        )
        return [dict(r) for r in await cur.fetchall()]

    # ---- contacts (seeded here, managed in phase 10) ----

    async def upsert_contact(
        self,
        *,
        account_id: str,
        contact_key: str,
        platform: str | None,
        name: str | None,
        username: str | None,
        tag: str,
    ) -> None:
        """Auto-seed/refresh a potential contact from a comment or DM
        author. Merges the new tag into the existing csv (dedup)."""
        conn = await self._db.connect()
        existing = await conn.execute(
            "SELECT tags FROM zernio_contacts "
            "WHERE account_id = ? AND contact_key = ?",
            (account_id, contact_key),
        )
        row = await existing.fetchone()
        tags = {t for t in ((row["tags"] or "").split(",") if row else []) if t}
        tags.add(tag)
        if platform:
            tags.add(platform)
        tags_csv = ",".join(sorted(tags))
        if row:
            await conn.execute(
                "UPDATE zernio_contacts SET tags = ?, name = COALESCE(?, name), "
                "username = COALESCE(?, username), last_seen = ? "
                "WHERE account_id = ? AND contact_key = ?",
                (tags_csv, name, username, _now(), account_id, contact_key),
            )
        else:
            await conn.execute(
                "INSERT INTO zernio_contacts "
                "(account_id, contact_key, platform, name, username, tags, "
                "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    account_id, contact_key, platform, name, username, tags_csv,
                    _now(), _now(),
                ),
            )
        await conn.commit()

    async def list_contacts(
        self, account_ids: list[str], *, tag: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        if not account_ids:
            return []
        placeholders = ",".join("?" for _ in account_ids)
        where = f"account_id IN ({placeholders})"
        params: list[object] = list(account_ids)
        if tag:
            where += " AND (',' || tags || ',') LIKE ?"
            params.append(f"%,{tag},%")
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT account_id, contact_key, platform, name, username, tags, "
            "zernio_contact_id, first_seen, last_seen FROM zernio_contacts "
            f"WHERE {where} ORDER BY last_seen DESC LIMIT ?",  # fixed frags, bound
            (*params, int(limit)),
        )
        return [dict(r) for r in await cur.fetchall()]


class ZernioCalendarRepo:
    """External (native) posts for the unified calendar (migration 036).

    Tenant-free at rest — keyed by (account_id, platform-native
    post_id) because post.external.* carries no profileId. The calendar
    route resolves to a tenant by matching account_id against the
    viewing tenant's connected accounts."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def upsert(
        self,
        *,
        account_id: str,
        post_id: str,
        platform: str | None,
        content: str | None,
        url: str | None,
        thumbnail_url: str | None,
        media_type: str | None,
        published_at: str | None,
    ) -> None:
        """Insert or refresh one external post. Idempotent: the
        first-sync `created` and any later `updated` both land here, and
        a re-delivered `created` overwrites identically. Clears any
        prior 'deleted' status (a post can reappear)."""
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO zernio_calendar "
            "(account_id, post_id, platform, content, url, thumbnail_url, "
            "media_type, published_at, status, deleted_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?) "
            "ON CONFLICT(account_id, post_id) DO UPDATE SET "
            "platform = excluded.platform, content = excluded.content, "
            "url = excluded.url, thumbnail_url = excluded.thumbnail_url, "
            "media_type = excluded.media_type, "
            "published_at = excluded.published_at, "
            "status = 'active', deleted_at = NULL, "
            "updated_at = excluded.updated_at",
            (
                account_id, post_id, platform, content, url, thumbnail_url,
                media_type, published_at, _now(),
            ),
        )
        await conn.commit()

    async def mark_deleted(
        self, *, account_id: str, post_id: str, deleted_at: str | None
    ) -> None:
        """Flag an external post removed on the platform. Keeps the row
        (the calendar greys it out) rather than dropping it."""
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE zernio_calendar SET status = 'deleted', "
            "deleted_at = ?, updated_at = ? "
            "WHERE account_id = ? AND post_id = ?",
            (deleted_at or _now(), _now(), account_id, post_id),
        )
        await conn.commit()

    async def list_for_accounts(
        self,
        account_ids: list[str],
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        include_deleted: bool = True,
    ) -> list[dict[str, Any]]:
        """External posts for a set of account ids (the tenant's), in a
        date window. Empty list for no accounts (no SQL injection of an
        empty IN ())."""
        if not account_ids:
            return []
        placeholders = ",".join("?" for _ in account_ids)
        where = f"account_id IN ({placeholders})"
        params: list[object] = list(account_ids)
        if date_from:
            where += " AND published_at >= ?"
            params.append(date_from)
        if date_to:
            where += " AND published_at <= ?"
            params.append(date_to)
        if not include_deleted:
            where += " AND status = 'active'"
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT account_id, post_id, platform, content, url, "
            "thumbnail_url, media_type, published_at, status, deleted_at "
            f"FROM zernio_calendar WHERE {where} "  # fixed fragments, params bound
            "ORDER BY published_at DESC",
            tuple(params),
        )
        return [dict(r) for r in await cur.fetchall()]


class ZernioPublishSnapshotsRepo:
    """Per-post daily metric snapshots (migration 035).

    Persist-only history for the future clip-selection feedback loop.
    tenant_id is explicit (the snapshot job iterates tenants); the
    UNIQUE (post_id, day) index makes a same-day re-run idempotent."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def upsert(
        self,
        tenant_id: str,
        *,
        post_id: str,
        day: str,
        metrics_json: str,
        platforms_json: str | None = None,
    ) -> None:
        """Insert or refresh the (post, day) snapshot."""
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO zernio_publish_snapshots "
            "(id, tenant_id, post_id, day, metrics_json, platforms_json, captured_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(post_id, day) DO UPDATE SET "
            "metrics_json = excluded.metrics_json, "
            "platforms_json = excluded.platforms_json, "
            "captured_at = excluded.captured_at",
            (
                new_id("snp"), tenant_id, post_id, day,
                metrics_json, platforms_json, _now(),
            ),
        )
        await conn.commit()

    async def latest_for_tenant(
        self, tenant_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Newest snapshot per post for a tenant (one row per post,
        most-recent day). Drives the internal analytics endpoint."""
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT post_id, day, metrics_json, platforms_json, captured_at "
            "FROM zernio_publish_snapshots WHERE tenant_id = ? "
            "AND day = (SELECT MAX(day) FROM zernio_publish_snapshots s2 "
            "           WHERE s2.post_id = zernio_publish_snapshots.post_id) "
            "ORDER BY day DESC LIMIT ?",
            (tenant_id, int(limit)),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def count_for_tenant(self, tenant_id: str) -> int:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT COUNT(*) AS n FROM zernio_publish_snapshots WHERE tenant_id = ?",
            (tenant_id,),
        )
        row = await cur.fetchone()
        return int(row["n"]) if row else 0


class ZernioAutoRetriesRepo:
    """Once-only guard for the post.failed auto-retry (migration 034).

    Tenant-free (the webhook boundary has no bound tenant); keyed by
    Zernio post id. `claim` is the dedup point — it returns False when
    a retry was already claimed for this post, so an at-least-once
    redelivery never schedules a second retry.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def claim(self, post_id: str, *, tenant_id: str | None) -> bool:
        """Atomically claim the single auto-retry for `post_id`. Returns
        True on a fresh claim, False if one already exists."""
        conn = await self._db.connect()
        cur = await conn.execute(
            "INSERT INTO zernio_auto_retries "
            "(post_id, tenant_id, attempted_at, outcome) "
            "VALUES (?, ?, ?, 'scheduled') "
            "ON CONFLICT DO NOTHING",
            (post_id, tenant_id, _now()),
        )
        await conn.commit()
        return cur.rowcount > 0

    async def set_outcome(self, post_id: str, *, outcome: str) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE zernio_auto_retries SET outcome = ? WHERE post_id = ?",
            (outcome, post_id),
        )
        await conn.commit()

    async def get(self, post_id: str) -> dict[str, Any] | None:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT post_id, tenant_id, attempted_at, outcome "
            "FROM zernio_auto_retries WHERE post_id = ?",
            (post_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row else None


class ZernioEventsRepo:
    """Inbound Zernio webhook event log (migration 031).

    Deliveries are at-least-once; `insert_dedup` is the dedup point
    (PRIMARY KEY on Zernio's stable event id). Rows keep the raw
    payload verbatim so later phases (calendar, inbox) can backfill
    their stores without re-asking Zernio. Tenant-free by design —
    webhooks are a server-to-server boundary and tenant resolution is
    exactly what the processor derives FROM these rows.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def insert_dedup(
        self,
        *,
        event_id: str,
        type: str,  # matches the column name
        payload: str,
        profile_id: str | None = None,
        tenant_id: str | None = None,
    ) -> bool:
        """Insert one event; return False when event_id already exists
        (a redelivery — the caller ACKs 200 without reprocessing)."""
        conn = await self._db.connect()
        cur = await conn.execute(
            "INSERT INTO zernio_events "
            "(event_id, type, payload, profile_id, tenant_id, received_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            (event_id, type, payload, profile_id, tenant_id, _now()),
        )
        await conn.commit()
        return cur.rowcount > 0

    async def get(self, event_id: str) -> ZernioEventRow | None:
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT event_id, type, payload, profile_id, tenant_id, "
            "received_at, processed, processed_at "
            "FROM zernio_events WHERE event_id = ?",
            (event_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["processed"] = bool(d.get("processed"))
        return ZernioEventRow.model_validate(d)

    async def mark_processed(
        self, event_id: str, *, tenant_id: str | None = None
    ) -> None:
        """Flip processed; also persist the resolved tenant when the
        processor figured one out (NULL stays NULL otherwise)."""
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE zernio_events SET processed = 1, processed_at = ?, "
            "tenant_id = COALESCE(?, tenant_id) WHERE event_id = ?",
            (_now(), tenant_id, event_id),
        )
        await conn.commit()

    async def list_unprocessed(self, limit: int = 100) -> list[ZernioEventRow]:
        """Oldest-first backlog — lets a sweep retry events whose
        background task died before mark_processed."""
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT event_id, type, payload, profile_id, tenant_id, "
            "received_at, processed, processed_at "
            "FROM zernio_events WHERE processed = 0 "
            "ORDER BY received_at ASC LIMIT ?",
            (int(limit),),
        )
        out: list[ZernioEventRow] = []
        for row in await cur.fetchall():
            d = dict(row)
            d["processed"] = bool(d.get("processed"))
            out.append(ZernioEventRow.model_validate(d))
        return out


_HUB_JOB_COLS = (
    "job_id, tenant_id, idempotency_key, source, mode, targets, video_url, "
    "title, caption, scheduled_for, zernio_post_id, status, platforms_json, "
    "error, created_at, updated_at"
)


class HubPublishJobsRepo:
    """Internal-API publish jobs (migration 032).

    The service API addresses tenants by id with a service token, so
    tenant_id is an EXPLICIT first argument here (not the bound-tenant
    context) — every query still filters on it. The two tenant-free
    methods are the webhook-processor joins, keyed by Zernio post id.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def create(
        self,
        tenant_id: str,
        *,
        source: str,
        mode: str,
        targets: list[str],
        video_url: str,
        title: str | None = None,
        caption: str | None = None,
        scheduled_for: str | None = None,
        idempotency_key: str | None = None,
    ) -> HubPublishJobRow:
        job_id = new_id("hpj")
        conn = await self._db.connect()
        await conn.execute(
            "INSERT INTO hub_publish_jobs "
            "(job_id, tenant_id, idempotency_key, source, mode, targets, "
            "video_url, title, caption, scheduled_for, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (
                job_id, tenant_id, idempotency_key, source, mode,
                ",".join(targets), video_url, title, caption,
                scheduled_for, _now(),
            ),
        )
        await conn.commit()
        job = await self.get(tenant_id, job_id)
        assert job is not None
        return job

    async def get(self, tenant_id: str, job_id: str) -> HubPublishJobRow | None:
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_HUB_JOB_COLS} FROM hub_publish_jobs "
            "WHERE job_id = ? AND tenant_id = ?",
            (job_id, tenant_id),
        )
        row = await cur.fetchone()
        return HubPublishJobRow.model_validate(dict(row)) if row else None

    async def find_by_idempotency(
        self, tenant_id: str, idempotency_key: str
    ) -> HubPublishJobRow | None:
        """The replay path: a repeated key returns the ORIGINAL job."""
        conn = await self._db.connect()
        cur = await conn.execute(
            f"SELECT {_HUB_JOB_COLS} FROM hub_publish_jobs "
            "WHERE tenant_id = ? AND idempotency_key = ?",
            (tenant_id, idempotency_key),
        )
        row = await cur.fetchone()
        return HubPublishJobRow.model_validate(dict(row)) if row else None

    async def set_zernio_post(
        self, tenant_id: str, job_id: str, *, post_id: str, status: str
    ) -> None:
        """Link the accepted Zernio post and move past 'pending'."""
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE hub_publish_jobs SET zernio_post_id = ?, status = ?, "
            "updated_at = ? WHERE job_id = ? AND tenant_id = ?",
            (post_id, status, _now(), job_id, tenant_id),
        )
        await conn.commit()

    async def set_error(
        self, tenant_id: str, job_id: str, *, error: str
    ) -> None:
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE hub_publish_jobs SET status = 'failed', error = ?, "
            "updated_at = ? WHERE job_id = ? AND tenant_id = ?",
            (error, _now(), job_id, tenant_id),
        )
        await conn.commit()

    async def set_status_by_post_id(
        self, post_id: str, *, status: str, platforms_json: str | None = None
    ) -> None:
        """Tenant-FREE webhook join: post.* events update the hub job
        the same way they update zernio_publishes. No-op for posts that
        didn't come through the internal API."""
        conn = await self._db.connect()
        await conn.execute(
            "UPDATE hub_publish_jobs SET status = ?, "
            "platforms_json = COALESCE(?, platforms_json), updated_at = ? "
            "WHERE zernio_post_id = ?",
            (status, platforms_json, _now(), post_id),
        )
        await conn.commit()

    async def get_tenant_for_post(self, post_id: str) -> str | None:
        """Tenant-FREE reverse lookup for the webhook processor."""
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT tenant_id FROM hub_publish_jobs WHERE zernio_post_id = ?",
            (post_id,),
        )
        row = await cur.fetchone()
        return str(row["tenant_id"]) if row else None

    async def count_for_day(
        self, tenant_id: str, *, platform: str, day: str
    ) -> int:
        """Hub posts targeting `platform` whose effective date (the
        scheduled date, or creation date for immediate posts) falls on
        `day` (YYYY-MM-DD, UTC). Drives the batch anti-spam cap."""
        conn = await self._db.connect()
        cur = await conn.execute(
            # substr(...,1,10) takes the YYYY-MM-DD prefix of the ISO-8601
            # timestamp string — portable across SQLite and Postgres (PG's
            # date() returns a date type, which breaks str param binding).
            "SELECT COUNT(*) AS n FROM hub_publish_jobs "
            "WHERE tenant_id = ? "
            "AND substr(COALESCE(scheduled_for, created_at), 1, 10) = ? "
            "AND (',' || targets || ',') LIKE ? "
            "AND status != 'failed'",
            (tenant_id, day, f"%,{platform},%"),
        )
        row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def recent_times_for_platform(
        self, tenant_id: str, *, platform: str, limit: int = 10
    ) -> list[str]:
        """Effective post times (ISO strings, newest-first) of the latest
        non-failed hub posts targeting `platform` — scheduled_for when set,
        else created_at. Feeds `next_best_time`'s min-gap check so best-time
        publishes never cluster tighter than the platform rulebook allows.
        ISO-8601 strings order lexicographically, so ORDER BY is portable."""
        conn = await self._db.connect()
        cur = await conn.execute(
            "SELECT COALESCE(scheduled_for, created_at) AS at "
            "FROM hub_publish_jobs "
            "WHERE tenant_id = ? "
            "AND (',' || targets || ',') LIKE ? "
            "AND status != 'failed' "
            "ORDER BY COALESCE(scheduled_for, created_at) DESC LIMIT ?",
            (tenant_id, f"%,{platform},%", int(limit)),
        )
        return [str(r["at"]) for r in await cur.fetchall() if r["at"]]
