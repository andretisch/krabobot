"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from krabobot.agent.hook import AgentHook, AgentHookContext
from krabobot.agent.tools.registry import ToolRegistry
from krabobot.providers.base import LLMProvider, ToolCallRequest
from krabobot.utils.helpers import build_assistant_message

_DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)
_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
DEFAULT_LOOP_DETECTED_MESSAGE = (
    "Stopped: the agent kept repeating the same tool calls without progress. "
    "Try rephrasing the task or breaking it into smaller steps."
)

# Anti-loop defaults (overridable via AgentRunSpec kwargs). Soft warn once, then hard stop.
LOOP_REPEAT_THRESHOLD = 3
LOOP_WINDOW_SIZE = 8
LOOP_WINDOW_THRESHOLD = 5
LOOP_PINGPONG_PAIRS = 3
LOOP_SAME_ERROR_THRESHOLD = 3

_LOOP_SOFT_MESSAGE = (
    "[Loop guard] Detected a repeating tool-call pattern ({reason}). "
    "Do not repeat the same tool with the same arguments. "
    "Change your approach, use a different tool, or answer the user without that call."
)

# OpenAI Chat Completions tool messages only allow string or *text* parts — not image_url.
# Tools that return native image blocks (read_file, web_fetch) must attach pixels via a bridge user message.
TOOL_VISION_BRIDGE_PREFIX = "[Служебно: зрение модели]"
_TOOL_VISION_BRIDGE_HEAD = (
    f"{TOOL_VISION_BRIDGE_PREFIX} Сообщения role=tool выше — только подтверждение вызова. "
    "Ниже в этом сообщении переданы реальные пиксели из инструментов (image_url). "
    "Опиши и отвечай по ним напрямую — не утверждай, что ты «не видишь файл на диске»."
)


def tool_call_signature(tool_call: ToolCallRequest) -> str:
    """Stable fingerprint: tool name + hash of canonical JSON arguments."""
    payload = json.dumps(
        tool_call.arguments,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256(
        f"{tool_call.name}\0{payload}".encode("utf-8"),
    ).hexdigest()[:16]
    return f"{tool_call.name}:{digest}"


def _result_error_text(result: Any) -> str | None:
    if isinstance(result, str) and result.startswith("Error"):
        return result
    return None


@dataclass(slots=True)
class _LoopGuard:
    """Detect repeated / alternating tool-call cycles during one agent run."""

    repeat_threshold: int = LOOP_REPEAT_THRESHOLD
    window_size: int = LOOP_WINDOW_SIZE
    window_threshold: int = LOOP_WINDOW_THRESHOLD
    pingpong_pairs: int = LOOP_PINGPONG_PAIRS
    same_error_threshold: int = LOOP_SAME_ERROR_THRESHOLD
    signatures: list[str] = field(default_factory=list)
    error_streak: list[tuple[str, str]] = field(default_factory=list)
    soft_intervened: bool = False

    def record(
        self,
        tool_calls: list[ToolCallRequest],
        results: list[Any],
    ) -> str | None:
        """Append this round and return a human-readable detection reason, if any."""
        for tool_call, result in zip(tool_calls, results):
            sig = tool_call_signature(tool_call)
            self.signatures.append(sig)
            err = _result_error_text(result)
            if err is not None:
                # Same tool name + same Error text (args may differ).
                key = (tool_call.name, err)
                if self.error_streak and self.error_streak[-1] == key:
                    self.error_streak.append(key)
                else:
                    self.error_streak = [key]
            else:
                self.error_streak = []
        return self.detect()

    def detect(self) -> str | None:
        reason = self._detect_repeat_streak()
        if reason:
            return reason
        reason = self._detect_window()
        if reason:
            return reason
        reason = self._detect_pingpong()
        if reason:
            return reason
        return self._detect_same_error()

    def _detect_repeat_streak(self) -> str | None:
        n = self.repeat_threshold
        if n <= 0 or len(self.signatures) < n:
            return None
        tail = self.signatures[-n:]
        if len(set(tail)) == 1:
            return f"identical tool call {n} times in a row ({tail[0]})"
        return None

    def _detect_window(self) -> str | None:
        window = self.window_size
        threshold = self.window_threshold
        if window <= 0 or threshold <= 0:
            return None
        recent = self.signatures[-window:]
        if len(recent) < threshold:
            return None
        sig, count = Counter(recent).most_common(1)[0]
        if count >= threshold:
            return f"tool call {sig} appeared {count} times in last {len(recent)} calls"
        return None

    def _detect_pingpong(self) -> str | None:
        pairs = self.pingpong_pairs
        need = pairs * 2
        if pairs <= 0 or len(self.signatures) < need:
            return None
        tail = self.signatures[-need:]
        a, b = tail[0], tail[1]
        if a == b:
            return None
        for i, sig in enumerate(tail):
            if sig != (a if i % 2 == 0 else b):
                return None
        return f"alternating tool calls {a} / {b} for {pairs} pairs"

    def _detect_same_error(self) -> str | None:
        n = self.same_error_threshold
        if n <= 0 or len(self.error_streak) < n:
            return None
        name, _err = self.error_streak[-1]
        return f"same Error result from {name} repeated {n} times"


def is_tool_vision_bridge_content(content: Any) -> bool:
    """True for synthetic user packets that carry tool image pixels to the LLM.

    These must not appear in the web transcript as human «Вы» bubbles.
    """
    if isinstance(content, str):
        return content.lstrip().startswith(TOOL_VISION_BRIDGE_PREFIX)
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") not in ("text", "input_text"):
                continue
            text = str(block.get("text") or "")
            if text.strip():
                return text.lstrip().startswith(TOOL_VISION_BRIDGE_PREFIX)
    return False


def _tool_result_contains_vision_media(result: Any) -> bool:
    if not isinstance(result, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "image_url"
        for b in result
    )


def _strip_block_meta(block: dict[str, Any]) -> dict[str, Any]:
    if "_meta" in block:
        return {k: v for k, v in block.items() if k != "_meta"}
    return dict(block)


def _vision_bridge_user_content_from_results(
    tool_calls: list[ToolCallRequest],
    results: list[Any],
) -> list[dict[str, Any]] | None:
    parts: list[dict[str, Any]] = []
    for tc, result in zip(tool_calls, results):
        if not _tool_result_contains_vision_media(result):
            continue
        if not isinstance(result, list):
            continue
        parts.append({
            "type": "text",
            "text": f"[Инструмент `{tc.name}` вернул изображение ниже.]",
        })
        for b in result:
            if isinstance(b, dict):
                parts.append(_strip_block_meta(b))
    if not parts:
        return None
    return [{"type": "text", "text": _TOOL_VISION_BRIDGE_HEAD}, *parts]


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    loop_detected_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    loop_guard: bool = True
    loop_repeat_threshold: int = LOOP_REPEAT_THRESHOLD
    loop_window_size: int = LOOP_WINDOW_SIZE
    loop_window_threshold: int = LOOP_WINDOW_THRESHOLD
    loop_pingpong_pairs: int = LOOP_PINGPONG_PAIRS
    loop_same_error_threshold: int = LOOP_SAME_ERROR_THRESHOLD


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def _turn_scope(self):
        """Share one TokenMap across all chat() calls in this run when anonymizing."""
        from krabobot.providers.anonymizing import AnonymizingProvider

        if isinstance(self.provider, AnonymizingProvider):
            from krabobot.agent.anonymize import token_map_scope

            return token_map_scope()
        return nullcontext()

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        loop_guard = (
            _LoopGuard(
                repeat_threshold=spec.loop_repeat_threshold,
                window_size=spec.loop_window_size,
                window_threshold=spec.loop_window_threshold,
                pingpong_pairs=spec.loop_pingpong_pairs,
                same_error_threshold=spec.loop_same_error_threshold,
            )
            if spec.loop_guard
            else None
        )

        with self._turn_scope():
            for iteration in range(spec.max_iterations):
                context = AgentHookContext(iteration=iteration, messages=messages)
                await hook.before_iteration(context)
                kwargs: dict[str, Any] = {
                    "messages": messages,
                    "tools": spec.tools.get_definitions(),
                    "model": spec.model,
                }
                if spec.temperature is not None:
                    kwargs["temperature"] = spec.temperature
                if spec.max_tokens is not None:
                    kwargs["max_tokens"] = spec.max_tokens
                if spec.reasoning_effort is not None:
                    kwargs["reasoning_effort"] = spec.reasoning_effort

                if hook.wants_streaming():
                    async def _stream(delta: str) -> None:
                        await hook.on_stream(context, delta)

                    response = await self.provider.chat_stream_with_retry(
                        **kwargs,
                        on_content_delta=_stream,
                    )
                else:
                    response = await self.provider.chat_with_retry(**kwargs)

                raw_usage = response.usage or {}
                usage = {
                    "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
                }
                context.response = response
                context.usage = usage
                context.tool_calls = list(response.tool_calls)

                if response.has_tool_calls:
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=True)

                    messages.append(build_assistant_message(
                        response.content or "",
                        tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    ))
                    tools_used.extend(tc.name for tc in response.tool_calls)

                    await hook.before_execute_tools(context)

                    results, new_events, fatal_error = await self._execute_tools(spec, response.tool_calls)
                    tool_events.extend(new_events)
                    context.tool_results = list(results)
                    context.tool_events = list(new_events)
                    if fatal_error is not None:
                        error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                        stop_reason = "tool_error"
                        context.error = error
                        context.stop_reason = stop_reason
                        await hook.after_iteration(context)
                        break

                    vision_user = _vision_bridge_user_content_from_results(
                        response.tool_calls,
                        results,
                    )
                    for tool_call, result in zip(response.tool_calls, results):
                        if _tool_result_contains_vision_media(result):
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": (
                                    f"[{tool_call.name}] Визуальный вывод (изображение) — "
                                    "пиксели в следующем сообщении пользователя (служебный мост для API)."
                                ),
                            })
                        else:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.name,
                                "content": result,
                            })
                    if vision_user:
                        messages.append({"role": "user", "content": vision_user})

                    if loop_guard is not None:
                        loop_reason = loop_guard.record(response.tool_calls, results)
                        if loop_reason:
                            if not loop_guard.soft_intervened:
                                soft = _LOOP_SOFT_MESSAGE.format(reason=loop_reason)
                                messages.append({"role": "user", "content": soft})
                                loop_guard.soft_intervened = True
                                logger.warning("Loop guard soft intervene: {}", loop_reason)
                            else:
                                stop_reason = "loop_detected"
                                final_content = (
                                    spec.loop_detected_message
                                    or DEFAULT_LOOP_DETECTED_MESSAGE
                                )
                                error = final_content
                                context.final_content = final_content
                                context.error = error
                                context.stop_reason = stop_reason
                                logger.warning("Loop guard hard stop: {}", loop_reason)
                                await hook.after_iteration(context)
                                break

                    await hook.after_iteration(context)
                    continue

                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=False)

                clean = hook.finalize_content(context, response.content)
                if response.finish_reason == "error":
                    final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                    stop_reason = "error"
                    error = final_content
                    context.final_content = final_content
                    context.error = error
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break

                messages.append(build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                ))
                final_content = clean
                context.final_content = final_content
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break
            else:
                stop_reason = "max_iterations"
                template = spec.max_iterations_message or _DEFAULT_MAX_ITERATIONS_MESSAGE
                final_content = template.format(max_iterations=spec.max_iterations)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
        )

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None]:
        if spec.concurrent_tools:
            tool_results = await asyncio.gather(*(
                self._run_tool(spec, tool_call)
                for tool_call in tool_calls
            ))
        else:
            tool_results = [
                await self._run_tool(spec, tool_call)
                for tool_call in tool_calls
            ]

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        for result, event, error in tool_results:
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        return results, events, fatal_error

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
    ) -> tuple[Any, dict[str, str], BaseException | None]:
        # Tool args are already decoded by AnonymizingProvider when anonymize is on.
        arguments = tool_call.arguments
        try:
            result = await spec.tools.execute(tool_call.name, arguments)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            return f"Error: {type(exc).__name__}: {exc}", event, None

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        return result, {
            "name": tool_call.name,
            "status": "error" if isinstance(result, str) and result.startswith("Error") else "ok",
            "detail": detail,
        }, None
