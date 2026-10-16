"""LLMRouter — single point through which every LLM call flows.

Why a router (per CLAUDE.md hard rule #3):
    - Centralized cost tracking: every call writes a row to the call log
      (Phase 0: JSONL on disk; Phase 1+: `llm_calls` table).
    - Provider fallback: try `primary`, then each `fallback` in order.
    - Retries with exponential backoff on transient failures.
    - Structured output validation: callers get a typed Pydantic model back,
      never raw JSON.

Anthropic is the only configured provider. The router still supports an
arbitrary fallback chain; if a future config introduces a second provider,
this code doesn't need to change — just register a factory entry below
and reference it in `routing.<purpose>.fallbacks`.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import functools
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from chalybclip.errors import BudgetExceeded, LLMError

from .config import LLMConfig, ProviderConfig, Quality
from .prices import cost_micros
from .provider import LLMProvider, MultimodalImage, ProviderResult, RetryableLLMError

if TYPE_CHECKING:
    from chalybclip.db import Database
    from chalybclip.governance import BudgetGovernor

T = TypeVar("T", bound=BaseModel)

_router_log = logging.getLogger("chalybclip.llm.router")

ProviderFactory = Callable[[str, ProviderConfig, str], LLMProvider | None]
ProviderInvoker = Callable[[LLMProvider, str], Awaitable[ProviderResult]]

# ---- billing circuit breaker -------------------------------------------
# A billing failure (credit balance exhausted, monthly usage cap hit) is not
# transient: every subsequent call fails identically until a human fixes the
# account. Without a breaker, one drained account produced 1,868 failed hook
# calls in two weeks — each one burning latency and log noise per clip.
# State is process-global (not per-router) because routers are constructed
# per pipeline run; a restart merely costs one failing call to re-trip it.
_BILLING_ERROR_MARKERS = (
    "credit balance", "usage limits", "billing", "insufficient credits",
    "quota exceeded",
)
_BILLING_LOCKOUT_S = 3600.0
_billing_lockouts: dict[str, _dt.datetime] = {}


def _is_billing_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _BILLING_ERROR_MARKERS)


def reset_billing_lockouts() -> None:
    """Clear breaker state — for tests and operator tooling."""
    _billing_lockouts.clear()


class CallLogRow(BaseModel):
    """One row appended to `llm_calls.jsonl` per `complete()` call."""

    ts: str
    tenant_id: str
    purpose: str
    provider: str
    model: str
    quality: Quality
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd_micros: int = 0
    status: str = "ok"
    error: str | None = None
    attempts: int = 1


def _default_provider_factory(
    name: str, config: ProviderConfig, api_key: str
) -> LLMProvider | None:
    """Construct configured providers; return None for unimplemented ones.

    Dispatch is on `config.kind`, so `llm.yaml` can register any number of
    entries per protocol (e.g. `ollama` and `ollama_vision` both
    openai_compatible) without touching router code.
    """
    kind = getattr(config, "kind", "anthropic")
    if kind == "anthropic" and name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key, config=config)
    if kind == "openai_compatible":
        from .openai_compatible_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider(api_key=api_key, config=config)
    return None


class LLMRouter:
    """Routes typed LLM calls through `provider chain → retry → validate → log`."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        api_keys: dict[str, str] | None = None,
        call_log_path: Path | None = None,
        db: Database | None = None,
        provider_factory: ProviderFactory | None = None,
        clock: Callable[[], _dt.datetime] | None = None,
        governor: BudgetGovernor | None = None,
    ):
        self._config = config
        self._api_keys = api_keys if api_keys is not None else _read_api_keys(config)
        self._call_log_path = call_log_path
        self._db = db
        self._provider_factory = provider_factory or _default_provider_factory
        self._clock = clock or (lambda: _dt.datetime.now(_dt.UTC))
        self._providers: dict[str, LLMProvider | None] = {}
        # Phase 2: optional pre-call gate. When set, every complete*() consults
        # `governor.check_llm_spend(tenant_id)` before issuing the request and
        # re-raises BudgetExceeded after emitting `llm.budget_exhausted`.
        self._governor = governor

    async def complete(
        self,
        *,
        tenant_id: str,
        purpose: str,
        system: str,
        user: str,
        schema: type[T],
        quality: Quality | None = None,
    ) -> T:
        """Run one text-only LLM completion with retries + provider fallback.

        Args:
            tenant_id: Owns the call (cost is attributed here).
            purpose: Routing key (e.g. `variant_generation`). Must exist in
                `config.routing`.
            system: System prompt (persona voice, instructions, etc.).
            user: User prompt - typically the clip context.
            schema: Pydantic model the response must validate against.
            quality: Override the routing rule's `default_quality`.
        """

        async def invoke(provider: LLMProvider, model: str) -> ProviderResult:
            return await provider.complete(
                tenant_id=tenant_id,
                model=model,
                system=system,
                user=user,
                schema=schema,
            )

        return await self._invoke(
            tenant_id=tenant_id,
            purpose=purpose,
            schema=schema,
            quality=quality,
            invoke_provider=invoke,
        )

    async def complete_multimodal(
        self,
        *,
        tenant_id: str,
        purpose: str,
        system: str,
        user: str,
        images: list[MultimodalImage],
        schema: type[T],
        quality: Quality | None = None,
    ) -> T:
        """Run one multimodal (text + images) completion.

        Same retry/fallback/cost-log path as `complete()` - the only
        difference is the provider call carries an `images` list. Phase 1
        ships local-bytes images (re-encoded to base64 inside the Anthropic
        provider); Phase 3 will extend this to S3 URLs without changing the
        router signature.
        """
        if not images:
            raise LLMError("complete_multimodal requires at least one image")

        async def invoke(provider: LLMProvider, model: str) -> ProviderResult:
            return await provider.complete_multimodal(
                tenant_id=tenant_id,
                model=model,
                system=system,
                user=user,
                images=images,
                schema=schema,
            )

        return await self._invoke(
            tenant_id=tenant_id,
            purpose=purpose,
            schema=schema,
            quality=quality,
            invoke_provider=invoke,
        )

    async def _invoke(
        self,
        *,
        tenant_id: str,
        purpose: str,
        schema: type[T],
        quality: Quality | None,
        invoke_provider: ProviderInvoker,
    ) -> T:
        """Run the provider chain for a single call (text or multimodal)."""
        rule = self._config.routing.get(purpose)
        if rule is None:
            raise LLMError(f"unknown routing purpose: {purpose}")
        effective_quality: Quality = quality or rule.default_quality

        # Pre-call budget gate (Phase 2 Task 1). Refuses cleanly if today's
        # tenant LLM spend already meets the daily ceiling. Emits an event
        # so the dashboard's spend cards know why traffic stopped.
        if self._governor is not None:
            try:
                await self._governor.check_llm_spend(tenant_id)
            except BudgetExceeded as e:
                await self._emit_event(
                    tenant_id=tenant_id,
                    type_="llm.budget_exhausted",
                    payload={"purpose": purpose, "error": str(e)},
                )
                raise

        provider_chain = [rule.primary, *rule.fallbacks]
        last_error: Exception | None = None
        # Collect every provider's failure reason so the final error message
        # surfaces the real first failure (e.g. anthropic key missing) instead
        # of just the last provider's "not available" — that bit me at 11am.
        chain_errors: list[str] = []

        for provider_name in provider_chain:
            lockout_until = _billing_lockouts.get(provider_name)
            if lockout_until is not None:
                if self._clock() < lockout_until:
                    reason = (
                        f"billing lockout active until {lockout_until.isoformat()} "
                        "(account out of credits / usage cap hit)"
                    )
                    last_error = LLMError(f"provider locked out: {provider_name} ({reason})")
                    chain_errors.append(f"{provider_name}: {reason}")
                    continue
                _billing_lockouts.pop(provider_name, None)
            provider = self._get_provider(provider_name)
            if provider is None:
                provider_cfg = self._config.providers.get(provider_name)
                has_key = bool(self._api_keys.get(provider_name))
                if provider_cfg is None:
                    reason = "no config block"
                elif provider_cfg.api_key_required and not has_key:
                    reason = f"{provider_cfg.api_key_env or '?'} env var missing/empty"
                else:
                    reason = "factory returned None (provider not implemented)"
                last_error = LLMError(f"provider not available: {provider_name} ({reason})")
                chain_errors.append(f"{provider_name}: {reason}")
                continue
            model = self._config.model_for(provider_name, effective_quality)

            try:
                validated, attempts, spent = await self._call_with_retries(
                    schema=schema,
                    invoke=functools.partial(invoke_provider, provider, model),
                )
            except (LLMError, ValidationError) as e:
                # Failed attempts still consumed tokens (schema-invalid
                # output, a response with no tool_use block) — bill them.
                failed_usage = _sum_usage(getattr(e, "spent", None) or [])
                await self._log(
                    tenant_id=tenant_id,
                    purpose=purpose,
                    provider=provider_name,
                    model=model,
                    quality=effective_quality,
                    usage=failed_usage,
                    cost_usd_micros=self._compute_cost_micros(
                        provider=provider_name, model=model, usage=failed_usage,
                    ),
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                    attempts=getattr(e, "attempts", 1),
                )
                last_error = e
                chain_errors.append(f"{provider_name}: {type(e).__name__}: {e}")
                if _is_billing_error(e):
                    until = self._clock() + _dt.timedelta(seconds=_BILLING_LOCKOUT_S)
                    _billing_lockouts[provider_name] = until
                    await self._emit_event(
                        tenant_id=tenant_id,
                        type_="llm.billing_lockout",
                        payload={
                            "provider": provider_name,
                            "purpose": purpose,
                            "until": until.isoformat(),
                            "error": str(e)[:300],
                        },
                    )
                # If there's another provider in the chain to try, emit
                # llm.fallback so dashboards can flag flaky primaries.
                if provider_name != provider_chain[-1]:
                    await self._emit_event(
                        tenant_id=tenant_id,
                        type_="llm.fallback",
                        payload={
                            "purpose": purpose,
                            "provider": provider_name,
                            "error": f"{type(e).__name__}: {e}",
                        },
                    )
                continue

            # Every attempt's tokens, not just the one that validated: a
            # retry after a schema violation paid for both responses.
            usage = _sum_usage(spent)
            cost = self._compute_cost_micros(
                provider=provider_name, model=model, usage=usage,
            )
            await self._log(
                tenant_id=tenant_id,
                purpose=purpose,
                provider=provider_name,
                model=model,
                quality=effective_quality,
                usage=usage,
                cost_usd_micros=cost,
                attempts=attempts,
            )
            return validated

        await self._emit_event(
            tenant_id=tenant_id,
            type_="llm.exhausted",
            payload={
                "purpose": purpose,
                "providers_tried": provider_chain,
                "error": f"{type(last_error).__name__}: {last_error}"
                if last_error is not None
                else None,
            },
        )
        chain_summary = " | ".join(chain_errors) if chain_errors else str(last_error)
        raise LLMError(
            f"all providers failed for purpose={purpose!r}: {chain_summary}"
        ) from last_error

    async def _call_with_retries(
        self,
        *,
        schema: type[T],
        invoke: Callable[[], Awaitable[ProviderResult]],
    ) -> tuple[T, int, list[ProviderResult]]:
        """Drive a single provider through `RetryConfig.max_attempts`.

        Returns `(validated, attempts, spent)` where `spent` is the usage of
        EVERY attempt that reached the provider — the billed ones that
        failed validation included. On failure the same list rides on the
        raised error as `.spent` so the error row is billed too."""
        retry = self._config.retry
        last_err: Exception | None = None
        spent: list[ProviderResult] = []
        for attempt in range(1, retry.max_attempts + 1):
            try:
                result = await invoke()
                spent.append(result)
                validated = schema.model_validate(result.output)
                return validated, attempt, spent
            except RetryableLLMError as e:
                if e.usage is not None:
                    spent.append(e.usage)
                last_err = e
                if attempt < retry.max_attempts:
                    backoff = retry.initial_backoff_s * (retry.backoff_multiplier ** (attempt - 1))
                    await asyncio.sleep(backoff)
                continue
            except ValidationError as e:
                # Treat schema violations as retryable - the LLM may produce a
                # better-formed object on the next attempt.
                last_err = e
                if attempt < retry.max_attempts:
                    backoff = retry.initial_backoff_s * (retry.backoff_multiplier ** (attempt - 1))
                    await asyncio.sleep(backoff)
                continue
            except LLMError as e:
                # Non-retryable: stop here, but keep what earlier attempts cost.
                e.attempts = attempt  # type: ignore[attr-defined]
                e.spent = spent  # type: ignore[attr-defined]
                raise

        # Annotate so the caller can record attempts in the failure log row.
        if last_err is not None:
            last_err.attempts = retry.max_attempts  # type: ignore[attr-defined]
            last_err.spent = spent  # type: ignore[attr-defined]
            raise last_err
        raise LLMError("retry loop exited without success or error")

    def _get_provider(self, name: str) -> LLMProvider | None:
        if name not in self._providers:
            cfg = self._config.providers.get(name)
            api_key = self._api_keys.get(name, "")
            # Keyless providers (self-hosted Ollama/vLLM) opt out via
            # api_key_required: false — an empty key is fine for them.
            if cfg is None or (cfg.api_key_required and not api_key):
                self._providers[name] = None
            else:
                self._providers[name] = self._provider_factory(name, cfg, api_key)
        return self._providers[name]

    def _compute_cost_micros(
        self, *, provider: str, model: str, usage: ProviderResult
    ) -> int:
        """Real USD micros incl. prompt-cache reads/writes. An unpriced
        model of a paid provider bills at the conservative (highest) rate
        and logs an error — never silently 0."""
        rates, known = self._config.rates_for(provider, model)
        if not known and _usage_total(usage) > 0:
            _router_log.error(
                "llm.price_unknown provider=%s model=%s — billed at the "
                "conservative rate; add it to chalybclip/llm/prices.py",
                provider, model,
            )
        return cost_micros(
            rates,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )

    async def _log(
        self,
        *,
        tenant_id: str,
        purpose: str,
        provider: str,
        model: str,
        quality: Quality,
        usage: ProviderResult | None = None,
        cost_usd_micros: int = 0,
        status: str = "ok",
        error: str | None = None,
        attempts: int = 1,
    ) -> None:
        """Write one cost-tracking row to the JSONL breadcrumb and the DB.

        Both writes are best-effort — the LLM call already happened, so
        a write failure here must not propagate up. The JSONL is the
        Phase 0 carry-over; the DB row is the Phase 1 source of truth.
        """
        usage = usage or ProviderResult(output={}, model=model)
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        ts = self._clock().isoformat()
        row = CallLogRow(
            ts=ts,
            tenant_id=tenant_id,
            purpose=purpose,
            provider=provider,
            model=model,
            quality=quality,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cost_usd_micros=cost_usd_micros,
            status=status,
            error=error,
            attempts=attempts,
        )

        if self._call_log_path is not None:
            try:
                self._call_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._call_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row.model_dump(), ensure_ascii=False))
                    f.write("\n")
            except OSError:
                pass

        if self._db is not None:
            from chalybclip.db import LLMCallsRepo
            from chalybclip.db.models import LLMCallRow
            from chalybclip.ids import new_id
            from chalybclip.tenancy import bound_tenant

            llm_call_id = new_id("llm")
            # Token T3 — attribute this cost to the current pipeline run's
            # stream, read from the structlog contextvar the pipeline binds
            # for the whole run (same source the step-event emitter uses).
            # None for non-pipeline calls.
            import structlog as _structlog
            _stream_id = _structlog.contextvars.get_contextvars().get("stream_id")
            db_row = LLMCallRow(
                id=llm_call_id,
                tenant_id=tenant_id,
                purpose=purpose,
                provider=provider,
                model=model,
                quality=quality,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd_micros=cost_usd_micros,
                status=status,
                error=error,
                attempts=attempts,
                ts=ts,
                stream_id=_stream_id,
            )
            try:
                with bound_tenant(tenant_id):
                    await LLMCallsRepo(self._db).record(db_row)
            except Exception:
                # Best-effort — the LLM call already happened and the JSONL
                # has the row; a DB write failure must not propagate up.
                pass

            # Slice NX.3 — usage back to Chalyb, via the durable outbox (the
            # drain delivers it; a restart no longer drops it). EVERY call
            # that consumed tokens is reported, failed/retried ones included
            # — the provider billed us for them. `amount` is the total of
            # all four token kinds; `metadata.tokens` carries the split.
            #
            # Operation tag: the LLM purpose, so /app/usage on the Chalyb
            # side can collapse one pipeline run into a single row.
            if _usage_total(usage) > 0 or cost_usd_micros > 0:
                from chalybclip.integrations.chalyb.outbox import enqueue_usage

                try:
                    await enqueue_usage(
                        self._db,
                        tenant_id=tenant_id,
                        kind="llm.tokens",
                        amount=_usage_total(usage),
                        cost_usd_micros=cost_usd_micros,
                        source_id=llm_call_id,
                        occurred_at_iso=ts,
                        provider=provider,
                        operation=purpose,
                        metadata={
                            "model": model,
                            "status": status,
                            "attempts": attempts,
                            "tokens": {
                                "input": usage.input_tokens,
                                "output": usage.output_tokens,
                                "cache_read": usage.cache_read_tokens,
                                "cache_write": usage.cache_write_tokens,
                            },
                        },
                    )
                except Exception:
                    _router_log.exception(
                        "llm usage enqueue failed · tenant=%s call=%s",
                        tenant_id, llm_call_id,
                    )

    async def _emit_event(
        self,
        *,
        tenant_id: str,
        type_: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append an llm.* event row when a DB is wired in. Best-effort."""
        if self._db is None:
            return
        from chalybclip.events import emit
        from chalybclip.tenancy import bound_tenant

        try:
            with bound_tenant(tenant_id):
                await emit(self._db, type_, payload)
        except Exception:
            pass


def _sum_usage(results: list[ProviderResult]) -> ProviderResult:
    """Token totals across attempts (output payload dropped)."""
    return ProviderResult(
        output={},
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
        cache_read_tokens=sum(r.cache_read_tokens for r in results),
        cache_write_tokens=sum(r.cache_write_tokens for r in results),
        model=results[-1].model if results else "",
    )


def _usage_total(usage: ProviderResult) -> int:
    return max(
        0,
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_read_tokens
        + usage.cache_write_tokens,
    )


def _read_api_keys(config: LLMConfig) -> dict[str, str]:
    """Pull each provider's API key out of the environment."""
    keys: dict[str, str] = {}
    for name, cfg in config.providers.items():
        keys[name] = os.environ.get(cfg.api_key_env, "")
    return keys
