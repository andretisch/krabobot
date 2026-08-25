"""Native Ollama HTTP provider for Ollama Cloud (https://ollama.com)."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

import httpx
import json_repair

from krabobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})


def is_ollama_cloud_base(api_base: str | None) -> bool:
    """Return True when api_base points at Ollama Cloud (ollama.com)."""
    if not api_base:
        return False
    host = (urlparse(api_base.strip()).hostname or "").lower()
    return host == "ollama.com" or host.endswith(".ollama.com")


def normalize_ollama_base(api_base: str) -> str:
    """Strip trailing slashes from an Ollama host URL."""
    return api_base.rstrip("/")


def resolve_ollama_api_key(api_key: str | None) -> str | None:
    """Resolve Ollama Cloud API key from config or OLLAMA_API_KEY env."""
    if api_key and api_key.strip():
        return api_key.strip()
    env_key = os.environ.get("OLLAMA_API_KEY")
    return env_key.strip() if env_key and env_key.strip() else None


class OllamaProvider(LLMProvider):
    """Direct Ollama HTTP API (/api/chat) for Ollama Cloud."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://ollama.com",
        default_model: str = "gemma4:cloud",
        timeout: float = 120.0,
    ):
        super().__init__(api_key=api_key, api_base=normalize_ollama_base(api_base))
        self.default_model = default_model
        self._timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"}

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

    def _build_body(
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
        body: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._to_ollama_messages(sanitized),
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max(1, max_tokens),
            },
        }
        if tools:
            body["tools"] = tools
        return body

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
        body = self._build_body(messages, tools, model, max_tokens, temperature, stream=False)
        url = f"{self.api_base}/api/chat"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=body, headers=self._headers)
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, dict):
                return LLMResponse(content="Error: invalid Ollama response.", finish_reason="error")
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
        body = self._build_body(messages, tools, model, max_tokens, temperature, stream=True)
        url = f"{self.api_base}/api/chat"
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        usage: dict[str, int] = {}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream("POST", url, json=body, headers=self._headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(chunk, dict):
                            continue

                        message = chunk.get("message") or {}
                        delta = message.get("content")
                        if isinstance(delta, str) and delta:
                            content_parts.append(delta)
                            if on_content_delta:
                                await on_content_delta(delta)

                        raw_tool_calls = message.get("tool_calls") or []
                        if raw_tool_calls:
                            tool_calls = self._parse_tool_calls(raw_tool_calls)

                        prompt_tokens = chunk.get("prompt_eval_count")
                        completion_tokens = chunk.get("eval_count")
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
