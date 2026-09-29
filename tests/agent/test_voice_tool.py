"""Tests for VoiceClientActionTool and pending-actions store."""

from __future__ import annotations

import pytest

from krabobot.agent.tools.voice import (
    VOICE_CLIENT_ACTIONS,
    VoiceClientActionTool,
    pop_voice_actions,
    queue_voice_action,
)


@pytest.fixture(autouse=True)
def _clear_pending():
    """Isolate pending store between tests."""
    pop_voice_actions("dev-1")
    pop_voice_actions("dev-2")
    yield
    pop_voice_actions("dev-1")
    pop_voice_actions("dev-2")


@pytest.mark.asyncio
async def test_queues_action_on_voice_channel() -> None:
    tool = VoiceClientActionTool()
    tool.set_context("voice", "dev-1")
    result = await tool.execute(action="meeting_stop")
    assert "meeting_stop" in result
    assert pop_voice_actions("dev-1") == [{"action": "meeting_stop"}]
    assert pop_voice_actions("dev-1") == []


@pytest.mark.asyncio
async def test_rejects_non_voice_channel() -> None:
    tool = VoiceClientActionTool()
    tool.set_context("telegram", "dev-1")
    result = await tool.execute(action="meeting_start")
    assert result.startswith("Error:")
    assert pop_voice_actions("dev-1") == []


@pytest.mark.asyncio
async def test_rejects_missing_chat_id() -> None:
    tool = VoiceClientActionTool()
    tool.set_context("voice", "")
    result = await tool.execute(action="end_dialog")
    assert "chat_id" in result
    assert pop_voice_actions("dev-1") == []


@pytest.mark.asyncio
async def test_all_action_enum_values() -> None:
    tool = VoiceClientActionTool()
    tool.set_context("voice", "dev-1")
    for action in VOICE_CLIENT_ACTIONS:
        await tool.execute(action=action)
    assert pop_voice_actions("dev-1") == [{"action": a} for a in VOICE_CLIENT_ACTIONS]


@pytest.mark.asyncio
async def test_unknown_action_rejected_by_schema_enum() -> None:
    tool = VoiceClientActionTool()
    errors = tool.validate_params({"action": "explode"})
    assert errors
    tool.set_context("voice", "dev-1")
    result = await tool.execute(action="explode")
    assert result.startswith("Error:")
    assert pop_voice_actions("dev-1") == []


def test_queue_and_pop_are_keyed_by_chat_id() -> None:
    queue_voice_action("dev-1", "meeting_start")
    queue_voice_action("dev-2", "end_dialog")
    assert pop_voice_actions("dev-1") == [{"action": "meeting_start"}]
    assert pop_voice_actions("dev-2") == [{"action": "end_dialog"}]
