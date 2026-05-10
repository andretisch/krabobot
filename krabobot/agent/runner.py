"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from krabobot.agent.hook import AgentHook, AgentHookContext
from krabobot.agent.tools.registry import ToolRegistry
from krabobot.providers.base import LLMProvider, ToolCallRequest
from krabobot.utils.helpers import build_assistant_message

_DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)
_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."

# OpenAI Chat Completions tool messages only allow string or *text* parts — not image_url.
# Tools that return native image blocks (read_file, web_fetch) must attach pixels via a bridge user message.
_TOOL_VISION_BRIDGE_HEAD = (
    "[Служебно: зрение модели] Сообщения role=tool выше — только подтверждение вызова. "
    "Ниже в этом сообщении переданы реальные пиксели из инструментов (image_url). "
    "Опиши и отвечай по ним напрямую — не утверждай, что ты «не видишь файл на диске»."
)


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
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False


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

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []

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
        try:
            result = await spec.tools.execute(tool_call.name, tool_call.arguments)
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
