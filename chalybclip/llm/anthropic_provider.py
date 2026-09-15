"""Anthropic Claude provider - structured output via tool use.

The structured-output trick: we declare a single `tool` whose `input_schema`
is the Pydantic model's JSON schema, then force `tool_choice` to that tool.
Claude returns a `tool_use` content block whose `input` is the structured
output dict, which the router validates against the originating schema.

Vision (`complete_multimodal`) wraps the same trick: the user message becomes
a content list mixing image blocks (base64-inlined) with the text prompt;
Claude still emits a single tool_use response.
"""

from __future__ import annotations

import base64
from typing import Any

import anthropic
from pydantic import BaseModel

from chalybclip.errors import LLMError

from .config import ProviderConfig
from .provider import LLMProvider, MultimodalImage, ProviderResult, RetryableLLMError

_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
_TOOL_NAME = "chalybclip_structured_output"


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
        tool = self._tool_for(schema)
        message = await self._send(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            tool=tool,
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
        tool = self._tool_for(schema)
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
            tool=tool,
        )
        return _parse_message(message, model=model)

    @staticmethod
    def _tool_for(schema: type[BaseModel]) -> dict[str, Any]:
        return {
            "name": _TOOL_NAME,
            "description": "Return the structured output requested by the system prompt.",
            "input_schema": schema.model_json_schema(),
        }

    async def _send(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tool: dict[str, Any],
    ) -> anthropic.types.Message:
        try:
            # The Anthropic SDK's overloads use TypedDicts for `tools` /
            # `tool_choice` / message content; passing plain dicts works at
            # runtime but doesn't satisfy the overload at type-check time.
            return await self._client.messages.create(  # type: ignore[call-overload, no-any-return]
                model=model,
                max_tokens=4096,
                system=system,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": _TOOL_NAME},
            )
        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status in _RETRYABLE_STATUSES:
                raise RetryableLLMError(f"anthropic {status}: {e}", status_code=status) from e
            raise LLMError(f"anthropic {status}: {e}") from e
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise RetryableLLMError(f"anthropic transient: {e}") from e


def _parse_message(message: anthropic.types.Message, *, model: str) -> ProviderResult:
    """Pull the tool_use input out of Claude's response."""
    structured: dict[str, Any] | None = None
    for block in message.content:
        if getattr(block, "type", None) == "tool_use":
            structured = dict(getattr(block, "input", {}) or {})
            break

    if structured is None:
        raise RetryableLLMError("anthropic returned no tool_use block")

    usage = message.usage
    return ProviderResult(
        output=structured,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        model=model,
    )
