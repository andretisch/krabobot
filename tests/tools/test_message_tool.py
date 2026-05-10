from unittest.mock import AsyncMock, MagicMock

import pytest

from krabobot.agent.tools.message import MessageTool


@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"


@pytest.mark.asyncio
async def test_message_tool_sets_tts_flag_from_resolver() -> None:
    resolver = MagicMock()
    resolver.lookup = AsyncMock(return_value=None)
    resolver.get_tts_enabled = AsyncMock(return_value=True)
    outbound: list = []

    async def capture(m):
        outbound.append(m)

    tool = MessageTool(send_callback=capture, user_resolver=resolver)
    tool.set_context("vk", "85554821", user_id="test-user-id")
    await tool.execute(content="Напоминание")

    assert len(outbound) == 1
    assert outbound[0].metadata["_tts_enabled_for_user"] is True
    resolver.get_tts_enabled.assert_awaited_once_with("test-user-id", default=False)
