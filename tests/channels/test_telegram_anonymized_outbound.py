"""Agent replies must reach Telegram when PII anonymize suppresses streaming."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from krabobot.agent.loop import AgentLoop
from krabobot.bus.events import InboundMessage
from krabobot.bus.queue import MessageBus
from krabobot.channels.manager import ChannelManager
from krabobot.channels.telegram import TelegramChannel, TelegramConfig
from krabobot.config.schema import ChannelsConfig
from krabobot.providers.base import GenerationSettings, LLMResponse

REPLY = "Привет! Да, я здесь."


class _FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []

    async def send_message(self, **kwargs):
        self.sent_messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.sent_messages))


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=512)

    async def chat_with_retry(**kwargs):
        return LLMResponse(content=REPLY, tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    return provider


@pytest.mark.asyncio
async def test_anonymized_agent_reply_is_sent_to_numeric_telegram_chat(tmp_path) -> None:
    """Streaming is off while anonymize is on; the final reply must still be delivered."""
    bus = MessageBus()
    loop = AgentLoop(
        bus=bus,
        provider=_provider(),
        workspace=tmp_path,
        model="test-model",
        anonymize=True,
    )
    await loop.user_resolver.link_account("owner", "telegram", "423648236")
    await loop.user_resolver.ensure_owner("owner")

    await loop._dispatch(
        InboundMessage(
            channel="telegram",
            sender_id="423648236",
            chat_id="423648236",
            content="Привит. Тут?",
            metadata={"_wants_stream": True, "message_id": 11},
        )
    )

    queued: list = []
    while True:
        try:
            queued.append(bus.outbound.get_nowait())
        except asyncio.QueueEmpty:
            break

    replies = [
        msg for msg in queued
        if msg.channel == "telegram" and msg.content and not msg.metadata.get("_progress")
    ]
    assert replies, "agent reply was not published"
    assert all(msg.chat_id == "423648236" for msg in replies)
    assert "|" not in replies[-1].chat_id

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", streaming=True),
        bus,
    )
    channel._app = SimpleNamespace(bot=_FakeBot())

    manager = ChannelManager.__new__(ChannelManager)
    manager.config = SimpleNamespace(channels=ChannelsConfig(send_max_retries=1))
    for msg in replies:
        await manager._send_with_retry(channel, msg)

    sent = channel._app.bot.sent_messages
    assert sent, "Telegram send was dropped"
    assert sent[-1]["chat_id"] == 423648236
    assert isinstance(sent[-1]["chat_id"], int)
    assert REPLY in sent[-1]["text"]
