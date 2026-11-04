"""Provider abstraction shared by all LLM backends.

The router doesn't know about vendor specifics — it just calls
`provider.complete(...)` (or `complete_multimodal(...)`) and gets back a
`ProviderResult`. New providers implement the `LLMProvider` protocol and the
router calls them via the `provider_factory` injected into its constructor.
Anthropic is the only built-in provider; the protocol exists so a future
addition (or a swap to a different vendor) doesn't touch router code.

Phase 1 adds the multimodal surface; Phase 0 only used `complete`. Providers
that don't support vision should raise `LLMError` from `complete_multimodal`;
the router treats that like any other provider failure and falls through to
the next entry in the chain.
"""

from __future__ import annotations

import functools
import hashlib
from typing import Any, Literal, Protocol, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chalybclip.errors import LLMError

from .config import Effort


class RetryableLLMError(LLMError):
    """A transient provider failure: 429, 5xx, network/timeout.

    The router retries up to `RetryConfig.max_attempts`. Non-retryable failures
    raise plain `LLMError` and skip straight to the next provider in the chain.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        usage: ProviderResult | None = None,
        raw_reply: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        # Tokens the failed attempt still consumed (e.g. a response with no
        # usable JSON). The router bills them like any other attempt.
        self.usage = usage
        # The model's reply can echo user transcripts (PII): kept for debugging,
        # never formatted into the message, which reaches the logs (rule 10).
        self.raw_reply = raw_reply


def reply_fingerprint(text: str) -> str:
    """Length + sha256 prefix: identifies a model reply in logs without its content."""
    return f"{len(text)} chars, sha256={hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


@functools.cache
def declared_field_names(schema: type[BaseModel]) -> frozenset[str]:
    """Every field name and alias declared anywhere in `schema`'s model graph."""
    names: set[str] = set()
    seen: set[type[BaseModel]] = set()

    def walk(tp: Any) -> None:
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            if tp in seen:
                return
            seen.add(tp)
            for name, field in tp.model_fields.items():
                names.add(name)
                if field.alias:
                    names.add(field.alias)
                walk(field.annotation)
        for arg in get_args(tp):  # list[X], dict[K, V], X | None, Annotated[...]
            walk(arg)

    walk(schema)
    return frozenset(names)


def _safe_loc(segment: str | int, known: frozenset[str]) -> str:
    # Extra fields and dict keys are chosen by the model and can carry PII.
    return str(segment) if isinstance(segment, int) or segment in known else "<key>"


class SchemaMismatchError(LLMError):
    """The reply parsed but failed the schema. The message names only field locations and
    error types; the pydantic error (which quotes model output) stays on `validation_error`."""

    def __init__(self, schema: type[BaseModel], error: ValidationError, raw_output: Any):
        known = declared_field_names(schema)
        problems = ", ".join(
            f"{'.'.join(_safe_loc(p, known) for p in item['loc']) or '<root>'}:{item['type']}"
            for item in error.errors(include_input=False, include_url=False)
        )
        super().__init__(
            f"{schema.__name__} validation failed ({error.error_count()} errors: {problems}; "
            f"{reply_fingerprint(repr(raw_output))})"
        )
        self.validation_error = error
        self.raw_output = raw_output


class TruncatedOutputError(LLMError):
    """The reply hit `max_tokens` before the JSON closed. Not retried (the same
    request would truncate again), but its tokens were billed: `usage` has them."""

    def __init__(self, message: str, *, usage: ProviderResult | None = None):
        super().__init__(message)
        self.usage = usage


class RefusalError(LLMError):
    """The model declined (`stop_reason: refusal`). Not retried, since the same
    request is declined again, but billed: `usage` has the tokens."""

    def __init__(
        self, message: str, *, category: str | None = None, usage: ProviderResult | None = None
    ):
        super().__init__(message)
        self.category = category
        self.usage = usage


class ProviderResult(BaseModel):
    """Raw structured output + token counts returned by a provider call."""

    output: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    # Anthropic prompt-cache split; `input_tokens` is UNCACHED input only, so the
    # four counts add up to everything billed.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str


class MultimodalImage(BaseModel):
    """One image attached to a multimodal LLM call.

    Phase 1 carries raw JPEG/PNG bytes; the provider does whatever encoding
    the wire format wants (base64 for Anthropic, multipart for others). Phase
    3 will add a remote-URL variant once the dashboard backs onto S3 - the
    provider switch on `media_type`/`url` will land then.
    """

    model_config = ConfigDict(extra="forbid")

    media_type: Literal["image/jpeg", "image/png", "image/webp"] = "image/jpeg"
    data: bytes = Field(description="Raw image bytes (JPEG/PNG/WebP).")


class LLMProvider(Protocol):
    """Minimal surface every provider must expose to the router."""

    async def complete(
        self,
        *,
        tenant_id: str,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        effort: Effort | None = None,
    ) -> ProviderResult: ...

    async def complete_multimodal(
        self,
        *,
        tenant_id: str,
        model: str,
        system: str,
        user: str,
        images: list[MultimodalImage],
        schema: type[BaseModel],
        effort: Effort | None = None,
    ) -> ProviderResult: ...
