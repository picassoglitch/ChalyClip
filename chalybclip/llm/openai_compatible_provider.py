"""Open-model provider — any OpenAI-compatible chat endpoint, no paid API.

Every mainstream open-LLM runtime speaks the OpenAI `/chat/completions`
dialect: Ollama (`http://localhost:11434/v1`), LM Studio, vLLM,
llama.cpp-server, and the hosted gateways (OpenRouter, Groq). This provider
targets that dialect over `httpx` directly — no vendor SDK, no new
dependency, and with a self-hosted runtime, no per-token bill.

Structured output: open runtimes don't share Anthropic's forced tool-use
trick, so we do it the portable way — the JSON schema is embedded in the
system prompt, `response_format: {"type": "json_object"}` nudges runtimes
that support JSON mode, and the response text is parsed defensively
(markdown fences and `<think>` blocks stripped, outermost object
extracted). A non-JSON reply raises `RetryableLLMError`, so the router's
retry loop asks again.

Vision: images ride as `data:` URLs in `image_url` content parts — the
OpenAI-compatible shape Ollama/vLLM accept for vision models (llava,
qwen-vl, llama-vision). Point a vision-capable model at the multimodal
purposes via its own provider entry in `config/llm.yaml`.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

import httpx
from pydantic import BaseModel

from chalybclip.errors import LLMError

from .config import ProviderConfig
from .provider import LLMProvider, MultimodalImage, ProviderResult, RetryableLLMError

_RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|```\s*$", re.MULTILINE)

_SCHEMA_INSTRUCTION = (
    "\n\nRespond with ONLY a single JSON object — no prose, no markdown "
    "fences, no commentary. The object MUST validate against this JSON "
    "Schema:\n{schema}"
)


class OpenAICompatibleProvider(LLMProvider):
    """`/chat/completions` provider for self-hosted / open-model runtimes."""

    def __init__(self, *, api_key: str, config: ProviderConfig):
        env_base = os.environ.get(config.base_url_env, "") if config.base_url_env else ""
        self._base_url = (env_base or config.base_url or "http://localhost:11434/v1").rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=config.timeout_s, headers=headers)
        self._json_mode = config.json_mode

    async def complete(
        self,
        *,
        tenant_id: str,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> ProviderResult:
        del tenant_id  # cost attribution happens in the router
        return await self._chat(
            model=model,
            system=system,
            schema=schema,
            user_content=user,
        )

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
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{img.media_type};base64,"
                        f"{base64.b64encode(img.data).decode('ascii')}"
                    )
                },
            }
            for img in images
        ]
        content.append({"type": "text", "text": user})
        return await self._chat(
            model=model,
            system=system,
            schema=schema,
            user_content=content,
        )

    async def _chat(
        self,
        *,
        model: str,
        system: str,
        schema: type[BaseModel],
        user_content: str | list[dict[str, Any]],
    ) -> ProviderResult:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "system",
                    "content": system + _SCHEMA_INSTRUCTION.format(schema=schema_json),
                },
                {"role": "user", "content": user_content},
            ],
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.post(f"{self._base_url}/chat/completions", json=body)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise RetryableLLMError(f"open-llm transient: {e}") from e

        if resp.status_code in _RETRYABLE_STATUSES:
            raise RetryableLLMError(
                f"open-llm {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )
        if resp.status_code >= 400:
            raise LLMError(f"open-llm {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
            text = str(data["choices"][0]["message"]["content"] or "")
        except (ValueError, LookupError, TypeError) as e:
            raise RetryableLLMError(f"open-llm malformed response envelope: {e}") from e

        structured = _extract_json_object(text)
        if structured is None:
            # Reasoning ramble / refusal / truncation — worth another attempt.
            raise RetryableLLMError(
                f"open-llm returned non-JSON content: {text[:200]!r}"
            )

        usage = data.get("usage") or {}
        return ProviderResult(
            output=structured,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            model=model,
        )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the outermost JSON object out of a model reply.

    Tolerates `<think>` reasoning blocks (DeepSeek-R1 family), markdown
    fences, and leading/trailing prose. Returns None when no parseable
    object exists."""
    cleaned = _FENCE_RE.sub("", _THINK_RE.sub("", text)).strip()
    for candidate in (cleaned, _outermost_braces(cleaned)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _outermost_braces(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else ""
