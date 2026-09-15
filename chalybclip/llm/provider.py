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

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from chalybclip.errors import LLMError


class RetryableLLMError(LLMError):
    """A transient provider failure: 429, 5xx, network/timeout.

    The router retries up to `RetryConfig.max_attempts`. Non-retryable failures
    raise plain `LLMError` and skip straight to the next provider in the chain.
    """

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ProviderResult(BaseModel):
    """Raw structured output + token counts returned by a provider call."""

    output: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
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
    ) -> ProviderResult: ...
