"""Tests for ExecTool background execution and completion callbacks."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from krabobot.agent.tools.shell import ExecTool
from krabobot.bus.events import InboundMessage
from krabobot.bus.queue import MessageBus
from krabobot.providers.base import GenerationSettings, LLMResponse


@pytest.mark.asyncio
async def test_background_exec_publishes_inbound_on_completion() -> None:
    bus = MessageBus()
    tool = ExecTool(bus=bus, timeout=60)
    tool.set_context("telegram", "chat-42", user_id="u1")

    result = await tool.execute(command="echo bg-done", background=True, label="echo-job")
    assert "started" in result
    assert "echo-job" in result

    msg = await asyncio.wait_for(bus.consume_inbound(), timeout=5.0)
    assert msg.channel == "system"
    assert msg.sender_id == "background_exec"
    assert msg.chat_id == "telegram:chat-42"
    assert msg.user_id == "u1"
    assert "echo-job" in msg.content
    assert "Exit code: 0" in msg.content
    assert "bg-done" in msg.content
    # Origin session key so AgentLoop locks /stop with the live conversation
    assert msg.session_key == "telegram:chat-42"
    assert msg.dispatch_key == "user:u1:telegram:chat-42"


@pytest.mark.asyncio
async def test_background_exec_non_zero_exit_announces_failure() -> None:
    bus = MessageBus()
    tool = ExecTool(bus=bus)
    tool.set_context("cli", "direct")

    # Cross-platform non-zero exit
    cmd = f'{sys.executable} -c "raise SystemExit(7)"'
    await tool.execute(command=cmd, background=True, label="fail-job")

    msg = await asyncio.wait_for(bus.consume_inbound(), timeout=5.0)
    assert "failed" in msg.content
    assert "Exit code: 7" in msg.content
    assert msg.chat_id == "cli:direct"
    assert msg.session_key == "cli:direct"


@pytest.mark.asyncio
async def test_background_exec_requires_bus() -> None:
    tool = ExecTool()
    result = await tool.execute(command="echo hi", background=True)
    assert "Error" in result
    assert "message bus" in result.lower()


@pytest.mark.asyncio
async def test_background_exec_still_applies_guards() -> None:
    bus = MessageBus()
    tool = ExecTool(bus=bus)
    result = await tool.execute(command="rm -rf /", background=True)
    assert "blocked" in result.lower()
    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_sync_exec_unchanged() -> None:
    tool = ExecTool(bus=MessageBus())
    result = await tool.execute(command="echo sync-ok")
    assert "sync-ok" in result
    assert "Exit code: 0" in result
    assert "started" not in result


@pytest.mark.asyncio
async def test_background_exec_cancel_by_session() -> None:
    bus = MessageBus()
    tool = ExecTool(bus=bus)
    tool.set_context("tg", "c1", user_id="u9")

    # Long-running process
    if sys.platform == "win32":
        cmd = f'{sys.executable} -c "import time; time.sleep(60)"'
    else:
        cmd = "sleep 60"

    await tool.execute(command=cmd, background=True, label="slow")
    assert tool.get_running_count() == 1

    count = await tool.cancel_by_session("user:u9:tg:c1")
    assert count == 1
    assert tool.get_running_count() == 0
    # Cancelled jobs should not publish a completion announce
    await asyncio.sleep(0.1)
    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_system_inbound_session_key_uses_origin() -> None:
    msg = InboundMessage(
        channel="system",
        sender_id="background_exec",
        chat_id="api:web-sess-1",
        content="done",
        user_id="owner",
    )
    assert msg.session_key == "api:web-sess-1"
    assert msg.dispatch_key == "user:owner:api:web-sess-1"


@pytest.mark.asyncio
async def test_background_exec_processed_by_agent_loop(tmp_path: Path) -> None:
    """Announce must be consumed by AgentLoop.run() and written into the origin session."""
    from krabobot.agent.loop import AgentLoop

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=64)
    replies = iter(
        [
            LLMResponse(content="Seed ack.", tool_calls=[]),
            LLMResponse(content="Announce handled.", tool_calls=[]),
        ]
    )

    async def _chat(**_kwargs):
        return next(replies)

    provider.chat_with_retry = AsyncMock(side_effect=_chat)
    provider.chat_stream_with_retry = AsyncMock(
        return_value=LLMResponse(content="unused", tool_calls=[])
    )

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        restrict_to_workspace=True,
    )
    loop.user_resolver.is_registered = AsyncMock(return_value=True)

    # Seed the api session the way serve/process_direct does (chat_id = session id)
    seed = await loop.process_direct(
        "start transcription",
        session_key="api:web-sess-1",
        channel="api",
        chat_id="web-sess-1",
        sender_id="web-sess-1",
    )
    assert seed is not None
    assert "Seed ack." in (seed.content or "")

    msg_stub = InboundMessage(
        channel="api", sender_id="web-sess-1", chat_id="web-sess-1", content=""
    )
    await loop._ensure_identity(msg_stub)
    runtime = await loop._runtime_for_message(msg_stub)

    exec_tool = runtime.tools.get("exec")
    assert isinstance(exec_tool, ExecTool)
    assert exec_tool.bus is bus
    assert exec_tool._origin_channel == "api"
    assert exec_tool._origin_chat_id == "web-sess-1"

    loop_task = asyncio.create_task(loop.run())
    try:
        await asyncio.sleep(0.05)
        result = await exec_tool.execute(
            command="echo transcript-ready",
            background=True,
            label="stt-job",
        )
        assert "started" in result.lower()

        for _ in range(50):
            session = runtime.sessions.get_or_create("api:web-sess-1")
            texts = [
                (m.get("content") or "")
                for m in session.messages
                if m.get("role") == "assistant"
            ]
            if any("Announce handled." in t for t in texts):
                break
            await asyncio.sleep(0.1)
        else:
            session = runtime.sessions.get_or_create("api:web-sess-1")
            pytest.fail(f"Announce was not processed; messages={session.messages!r}")

        assert provider.chat_with_retry.await_count >= 2
    finally:
        loop.stop()
        loop_task.cancel()
        try:
            await asyncio.wait_for(loop_task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
