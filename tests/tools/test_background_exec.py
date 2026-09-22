"""Tests for ExecTool background execution and completion callbacks."""

from __future__ import annotations

import asyncio
import sys

import pytest

from krabobot.agent.tools.shell import ExecTool
from krabobot.bus.queue import MessageBus


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
