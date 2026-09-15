"""Fake LLM provider - records calls, replays canned outputs / errors."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from chalybclip.errors import LLMError
from chalybclip.llm import LLMProvider, MultimodalImage, ProviderResult, RetryableLLMError


class FakeProvider(LLMProvider):
    """Replays a queue of (`output`/`exception`) responses in order."""

    def __init__(self, name: str = "fake"):
        self.name = name
        self.calls: list[dict[str, Any]] = []
        self._responses: list[ProviderResult | Exception] = []

    def queue_success(
        self,
        output: dict[str, Any],
        *,
        input_tokens: int = 100,
        output_tokens: int = 50,
        model: str = "fake-model",
    ) -> None:
        self._responses.append(
            ProviderResult(
                output=output,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
            )
        )

    def queue_retryable(self, message: str = "503", *, status_code: int = 503) -> None:
        self._responses.append(RetryableLLMError(message, status_code=status_code))

    def queue_fatal(self, message: str = "401") -> None:
        self._responses.append(LLMError(message))

    async def complete(
        self,
        *,
        tenant_id: str,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> ProviderResult:
        self.calls.append(
            {
                "kind": "text",
                "tenant_id": tenant_id,
                "model": model,
                "system": system,
                "user": user,
                "schema": schema.__name__,
            }
        )
        return self._next()

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
        self.calls.append(
            {
                "kind": "multimodal",
                "tenant_id": tenant_id,
                "model": model,
                "system": system,
                "user": user,
                "images": list(images),
                "n_images": len(images),
                "schema": schema.__name__,
            }
        )
        return self._next()

    def _next(self) -> ProviderResult:
        if not self._responses:
            raise AssertionError(f"FakeProvider({self.name}): no queued response")
        nxt = self._responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt
