"""Ollama provider with auto local/cloud detection via the official ollama client."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
import json_repair
from ollama import AsyncClient

from krabobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})

LOCAL_OLLAMA_HOST = "http://localhost:11434"
CLOUD_OLLAMA_HOST = "https://ollama.com"
PROBE_TIMEOUT = 1.5


def is_ollama_cloud_base(api_base: str | None) -> bool:
    """Return True when api_base points at Ollama Cloud (ollama.com)."""
    if not api_base:
        return False
    host = (urlparse(api_base.strip()).hostname or "").lower()
    return host == "ollama.com" or host.endswith(".ollama.com")


def normalize_ollama_host(api_base: str) -> str:
    """Normalize an Ollama host URL for the native client (no /v1 suffix)."""
    base = api_base.strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def resolve_ollama_api_key(api_key: str | None) -> str | None:
    """Resolve Ollama Cloud API key from config or OLLAMA_API_KEY env."""
    if api_key and api_key.strip():
        return api_key.strip()
    env_key = os.environ.get("OLLAMA_API_KEY")
    return env_key.strip() if env_key and env_key.strip() else None


def probe_local_ollama(
    host: str = LOCAL_OLLAMA_HOST,
    timeout: float = PROBE_TIMEOUT,
) -> bool:
    """Return True when a local Ollama daemon responds on /api/tags."""
    url = f"{normalize_ollama_host(host)}/api/tags"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            return response.status_code == 200
    except Exception:
        return False


@dataclass(frozen=True)
class OllamaConnection:
    """Resolved Ollama endpoint (explicit config or auto-detected)."""

    host: str
    api_key: str | None
    is_cloud: bool
    auto_detected: bool


def resolve_ollama_connection(
    explicit_api_base: str | None,
    api_key: str | None,
    *,
    probe: Callable[[], bool] | None = None,
) -> OllamaConnection:
    """Resolve Ollama host: explicit apiBase wins, else probe local then cloud."""
    resolved_key = resolve_ollama_api_key(api_key)

    if explicit_api_base and explicit_api_base.strip():
        host = normalize_ollama_host(explicit_api_base)
        is_cloud = is_ollama_cloud_base(host)
        return OllamaConnection(
            host=host,
            api_key=resolved_key if is_cloud else None,
            is_cloud=is_cloud,
            auto_detected=False,
        )

    probe_fn = probe or probe_local_ollama
    if probe_fn():
        return OllamaConnection(
            host=LOCAL_OLLAMA_HOST,
            api_key=None,
            is_cloud=False,
            auto_detected=True,
        )

    return OllamaConnection(
        host=CLOUD_OLLAMA_HOST,
        api_key=resolved_key,
        is_cloud=True,
        auto_detected=True,
    )


class OllamaProvider(LLMProvider):
    """Ollama chat via the official ollama Python client (local or cloud)."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str = LOCAL_OLLAMA_HOST,
        default_model: str = "gemma4:cloud",
        timeout: float = 120.0,
    ):
        host = normalize_ollama_host(api_base)
        super().__init__(api_key=api_key, api_base=host)
        self.default_model = default_model
        self._timeout = timeout
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = AsyncClient(
            host=host,
            headers=headers or None,
            timeout=timeout,
        )

    @staticmethod
    def _to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style messages to Ollama /api/chat format."""
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            clean: dict[str, Any] = {"role": role}
            content = msg.get("content")
            if content is not None:
                clean["content"] = content

            if role == "assistant" and msg.get("tool_calls"):
                tool_calls = []
                for index, tc in enumerate(msg["tool_calls"]):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        args = json_repair.loads(args) if args else {}
                    tool_calls.append({
                        "type": "function",
                        "function": {
                            "index": index,
                            "name": fn.get("name", ""),
                            "arguments": args if isinstance(args, dict) else {},
                        },
                    })
                if tool_calls:
                    clean["tool_calls"] = tool_calls

            if role == "tool":
                tool_name = msg.get("name") or msg.get("tool_name")
                if tool_name:
                    clean["tool_name"] = tool_name

            converted.append(clean)
        return converted

    def _build_chat_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> dict[str, Any]:
        sanitized = LLMProvider._sanitize_request_messages(
            self._sanitize_empty_content(messages),
            _ALLOWED_MSG_KEYS,
        )
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._to_ollama_messages(sanitized),
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max(1, max_tokens),
            },
        }
        if tools:
            kwargs["tools"] = tools
        return kwargs

    @staticmethod
    def _payload_from_response(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        return dict(payload)

    @staticmethod
    def _parse_tool_calls(raw_tool_calls: list[Any]) -> list[ToolCallRequest]:
        parsed: list[ToolCallRequest] = []
        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json_repair.loads(args) if args else {}
            parsed.append(ToolCallRequest(
                id=str(tc.get("id") or fn.get("name") or "tool"),
                name=str(fn.get("name") or ""),
                arguments=args if isinstance(args, dict) else {},
            ))
        return parsed

    def _parse_response(self, payload: dict[str, Any]) -> LLMResponse:
        message = payload.get("message") or {}
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            content = str(content)

        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = self._parse_tool_calls(raw_tool_calls) if raw_tool_calls else []
        finish_reason = "tool_calls" if tool_calls else "stop"

        usage: dict[str, int] = {}
        prompt_tokens = payload.get("prompt_eval_count")
        completion_tokens = payload.get("eval_count")
        if prompt_tokens is not None or completion_tokens is not None:
            prompt = int(prompt_tokens or 0)
            completion = int(completion_tokens or 0)
            usage = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _handle_error(exc: Exception) -> LLMResponse:
        return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")

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
        del reasoning_effort, tool_choice  # not supported on native Ollama API yet
        kwargs = self._build_chat_kwargs(messages, tools, model, max_tokens, temperature, stream=False)
        try:
            response = await self._client.chat(**kwargs)
            payload = self._payload_from_response(response)
            return self._parse_response(payload)
        except Exception as exc:
            return self._handle_error(exc)

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
        del reasoning_effort, tool_choice
        kwargs = self._build_chat_kwargs(messages, tools, model, max_tokens, temperature, stream=True)
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        usage: dict[str, int] = {}

        try:
            stream = await self._client.chat(**kwargs)
            async for chunk in stream:
                payload = self._payload_from_response(chunk)
                message = payload.get("message") or {}
                delta = message.get("content")
                if isinstance(delta, str) and delta:
                    content_parts.append(delta)
                    if on_content_delta:
                        await on_content_delta(delta)

                raw_tool_calls = message.get("tool_calls") or []
                if raw_tool_calls:
                    tool_calls = self._parse_tool_calls(raw_tool_calls)

                prompt_tokens = payload.get("prompt_eval_count")
                completion_tokens = payload.get("eval_count")
                if prompt_tokens is not None or completion_tokens is not None:
                    prompt = int(prompt_tokens or 0)
                    completion = int(completion_tokens or 0)
                    usage = {
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": prompt + completion,
                    }

            finish_reason = "tool_calls" if tool_calls else "stop"
            return LLMResponse(
                content="".join(content_parts) or None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
            )
        except Exception as exc:
            return self._handle_error(exc)

    def get_default_model(self) -> str:
        return self.default_model
