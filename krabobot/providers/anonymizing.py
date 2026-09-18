"""Provider wrapper that anonymizes all LLM-bound traffic.

Single choke point: every ``chat`` / ``chat_stream`` (and retry helpers) encodes
messages toward the model and decodes assistant content + tool-call arguments
before returning. Turn-local ``TokenMap`` is shared via ContextVar when a
``token_map_scope`` is active (AgentRunner); one-shot callers get an ephemeral map.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from krabobot.agent.anonymize import (
    TokenMap,
    anonymize_messages,
    decode_tool_arguments,
    get_active_token_map,
)
from krabobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


def wrap_provider_if_anonymize(provider: LLMProvider, enabled: bool) -> LLMProvider:
    """Return *provider* wrapped for PII masking when *enabled* (idempotent)."""
    if not enabled:
        return provider
    if isinstance(provider, AnonymizingProvider):
        return provider
    return AnonymizingProvider(provider)


class AnonymizingProvider(LLMProvider):
    """Decorator around an ``LLMProvider`` that masks PII on every LLM call."""

    def __init__(self, inner: LLMProvider):
        # Do not call LLMProvider.__init__ — forward state to the inner provider.
        self._inner = inner

    @property
    def inner(self) -> LLMProvider:
        """The wrapped provider (for isinstance checks in tests/diagnostics)."""
        return self._inner

    @property
    def api_key(self) -> str | None:
        return self._inner.api_key

    @api_key.setter
    def api_key(self, value: str | None) -> None:
        self._inner.api_key = value

    @property
    def api_base(self) -> str | None:
        return self._inner.api_base

    @api_base.setter
    def api_base(self, value: str | None) -> None:
        self._inner.api_base = value

    @property
    def generation(self):
        return self._inner.generation

    @generation.setter
    def generation(self, value) -> None:
        self._inner.generation = value

    def get_default_model(self) -> str:
        return self._inner.get_default_model()

    def _resolve_token_map(self) -> TokenMap:
        active = get_active_token_map()
        return active if active is not None else TokenMap()

    def _encode_messages(self, messages: list[dict[str, Any]], token_map: TokenMap) -> list[dict[str, Any]]:
        encoded = anonymize_messages(messages, token_map)
        if token_map.token_to_original:
            logger.debug(
                "PII anonymize: {} token(s) toward LLM ({})",
                len(token_map.token_to_original),
                ", ".join(
                    sorted({t.split("-")[0].strip("[]") for t in token_map.token_to_original})
                ),
            )
        return encoded

    def _decode_response(self, response: LLMResponse, token_map: TokenMap) -> LLMResponse:
        if not token_map.token_to_original:
            return response
        content = response.content
        if isinstance(content, str):
            content = token_map.decode(content)
        tool_calls = [
            ToolCallRequest(
                id=tc.id,
                name=tc.name,
                arguments=decode_tool_arguments(tc.arguments, token_map),
                extra_content=tc.extra_content,
                provider_specific_fields=tc.provider_specific_fields,
                function_provider_specific_fields=tc.function_provider_specific_fields,
            )
            for tc in response.tool_calls
        ]
        reasoning = response.reasoning_content
        if isinstance(reasoning, str):
            reasoning = token_map.decode(reasoning)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.finish_reason,
            usage=response.usage,
            reasoning_content=reasoning,
            thinking_blocks=response.thinking_blocks,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        token_map = self._resolve_token_map()
        encoded = self._encode_messages(messages, token_map)
        response = await self._inner.chat(
            messages=encoded,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        return self._decode_response(response, token_map)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        # Streaming would leak tokenized fragments before decode — buffer then decode.
        token_map = self._resolve_token_map()
        encoded = self._encode_messages(messages, token_map)
        response = await self._inner.chat(
            messages=encoded,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        decoded = self._decode_response(response, token_map)
        if on_content_delta and decoded.content:
            await on_content_delta(decoded.content)
        return decoded

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        temperature: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Encode → inner retry → decode (so mocks of chat_with_retry still see tokens)."""
        token_map = self._resolve_token_map()
        encoded = self._encode_messages(messages, token_map)
        kwargs: dict[str, Any] = {
            "messages": encoded,
            "tools": tools,
            "model": model,
            "tool_choice": tool_choice,
        }
        if max_tokens is not self._SENTINEL:
            kwargs["max_tokens"] = max_tokens
        if temperature is not self._SENTINEL:
            kwargs["temperature"] = temperature
        if reasoning_effort is not self._SENTINEL:
            kwargs["reasoning_effort"] = reasoning_effort
        response = await self._inner.chat_with_retry(**kwargs)
        return self._decode_response(response, token_map)

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        temperature: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        token_map = self._resolve_token_map()
        encoded = self._encode_messages(messages, token_map)
        # Non-streaming path: avoid leaking tokens via deltas.
        kwargs: dict[str, Any] = {
            "messages": encoded,
            "tools": tools,
            "model": model,
            "tool_choice": tool_choice,
        }
        if max_tokens is not self._SENTINEL:
            kwargs["max_tokens"] = max_tokens
        if temperature is not self._SENTINEL:
            kwargs["temperature"] = temperature
        if reasoning_effort is not self._SENTINEL:
            kwargs["reasoning_effort"] = reasoning_effort
        response = await self._inner.chat_with_retry(**kwargs)
        decoded = self._decode_response(response, token_map)
        if on_content_delta and decoded.content:
            await on_content_delta(decoded.content)
        return decoded
