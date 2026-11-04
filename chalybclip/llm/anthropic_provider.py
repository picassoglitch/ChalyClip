"""Anthropic Claude provider - structured output via `output_config.format`.

Each call sends the Pydantic model's JSON schema as a `json_schema` output
format, so Claude's reply is a text block holding JSON that matches it. The
router still validates that dict against the originating schema.

Forced tool use (`tool_choice` "tool"/"any") was the old trick; Opus 5.5,
Sonnet 5.5 and Fable 5.1 reject it with a 400, and structured outputs work on
every model we route (Haiku 4.5/5.5, Sonnet 5/5.5, Opus 5/5.5).

Vision (`complete_multimodal`) sends the same format: the user message becomes
a content list mixing image blocks (base64-inlined) with the text prompt.
"""

from __future__ import annotations

import base64
import functools
import json
from typing import Any

import anthropic
from pydantic import BaseModel

from chalybclip.errors import LLMError

from .config import ProviderConfig
from .provider import (
    LLMProvider,
    MultimodalImage,
    ProviderResult,
    RetryableLLMError,
    TruncatedOutputError,
)

_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


@functools.cache
def output_format_for(schema: type[BaseModel]) -> dict[str, Any]:
    """`output_config.format` for a schema. The SDK's transform drops the
    keywords the API can't enforce (min/max, lengths) into the description and
    sets `additionalProperties: false`; the router re-validates them."""
    return {"type": "json_schema", "schema": anthropic.transform_schema(schema)}


class AnthropicProvider(LLMProvider):
    """Concrete Anthropic provider implementing the `LLMProvider` protocol."""

    def __init__(self, *, api_key: str, config: ProviderConfig):
        self._config = config
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=config.base_url or None,
            timeout=config.timeout_s,
        )

    async def complete(
        self,
        *,
        tenant_id: str,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> ProviderResult:
        del tenant_id  # used by the router for logging, not the SDK call
        message = await self._send(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            schema=schema,
        )
        return _parse_message(message, model=model)

    async def complete_multimodal(
        self,
        *,
        tenant_id: str,
        model: str,
        system: str,
        user: str,
        images: list[MultimodalImage],
        schema: type[BaseModel],
    ) -> ProviderResult:
        del tenant_id
        # Image blocks first, then the text prompt - matches Anthropic's
        # documented best practice for multi-image requests.
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": base64.b64encode(img.data).decode("ascii"),
                },
            }
            for img in images
        ]
        content.append({"type": "text", "text": user})
        message = await self._send(
            model=model,
            system=system,
            messages=[{"role": "user", "content": content}],
            schema=schema,
        )
        return _parse_message(message, model=model)

    def _output_config(self, model: str, schema: type[BaseModel]) -> dict[str, Any]:
        config: dict[str, Any] = {"format": output_format_for(schema)}
        effort = self._config.effort.get(model)
        if effort is not None:
            config["effort"] = effort
        return config

    async def _send(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
    ) -> anthropic.types.Message:
        try:
            # The SDK's overloads use TypedDicts for message content and
            # `output_config`; plain dicts work at runtime but not at type-check.
            return await self._client.messages.create(  # type: ignore[call-overload, no-any-return]
                model=model,
                max_tokens=self._config.max_output_tokens,
                system=system,
                messages=messages,
                output_config=self._output_config(model, schema),
            )
        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status in _RETRYABLE_STATUSES:
                raise RetryableLLMError(f"anthropic {status}: {e}", status_code=status) from e
            raise LLMError(f"anthropic {status}: {e}") from e
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise RetryableLLMError(f"anthropic transient: {e}") from e


def _parse_message(message: anthropic.types.Message, *, model: str) -> ProviderResult:
    """Pull the JSON object out of Claude's structured-output reply.

    A reply cut off at `max_tokens` raises TruncatedOutputError (not retried).
    A refusal or text that isn't a JSON object raises RetryableLLMError. Both
    carry this attempt's usage so the router bills the tokens."""
    usage = message.usage
    result = ProviderResult(
        output={},
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        model=model,
    )
    stop = getattr(message, "stop_reason", None)
    if stop == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None)
        raise RetryableLLMError(f"anthropic refused (category={category})", usage=result)

    if stop == "max_tokens":
        # Thinking counts against max_tokens too; retrying the same request
        # would truncate again, so fail once and bill it.
        raise TruncatedOutputError(
            f"anthropic reply hit max_tokens ({result.output_tokens} output tokens)",
            usage=result,
        )

    text = "".join(
        getattr(block, "text", "") or ""
        for block in message.content
        if getattr(block, "type", None) == "text"
    )
    try:
        structured = json.loads(text)
    except json.JSONDecodeError:
        structured = None
    if not isinstance(structured, dict):
        raise RetryableLLMError(
            f"anthropic reply is not a JSON object (stop_reason={stop}, {len(text)} chars)",
            usage=result,
        )
    return result.model_copy(update={"output": structured})
