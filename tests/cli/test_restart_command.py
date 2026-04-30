"""Tests for /restart slash command."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from krabobot.bus.events import InboundMessage, OutboundMessage
from krabobot.providers.base import LLMResponse


def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from krabobot.agent.loop import AgentLoop
    from krabobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("krabobot.agent.loop.ContextBuilder"), \
         patch("krabobot.agent.loop.SessionManager"), \
         patch("krabobot.agent.loop.SubagentManager"):
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    loop.user_resolver.resolve_or_create = AsyncMock(return_value="u-test-owner")
    loop.user_resolver.ensure_owner = AsyncMock(return_value="u-test-owner")
    loop.user_resolver.is_owner = AsyncMock(return_value=True)
    return loop, bus


class TestRestartCommand:

    @pytest.mark.asyncio
    async def test_restart_sends_message_and_calls_execv(self):
        from krabobot.command.builtin import cmd_restart
        from krabobot.command.router import CommandContext

        loop, bus = _make_loop()
        loop.user_resolver.is_owner = AsyncMock(return_value=True)
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/restart")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

        with patch("krabobot.command.builtin.os.execv") as mock_execv:
            out = await cmd_restart(ctx)
            assert "Перезапускаюсь" in out.content

            await asyncio.sleep(1.5)
            mock_execv.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_intercepted_in_run_loop(self):
        """Verify /restart is handled at the run-loop level, not inside _dispatch."""
        loop, bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/restart")

        with patch.object(loop, "_dispatch", new_callable=AsyncMock) as mock_dispatch, \
             patch("krabobot.command.builtin.os.execv"):
            await bus.publish_inbound(msg)

            loop._running = True
            run_task = asyncio.create_task(loop.run())
            await asyncio.sleep(0.1)
            loop._running = False
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

            mock_dispatch.assert_not_called()
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert "Перезапускаюсь" in out.content

    @pytest.mark.asyncio
    async def test_restart_denied_for_non_owner(self):
        from krabobot.command.builtin import cmd_restart
        from krabobot.command.router import CommandContext

        loop, _bus = _make_loop()
        loop.user_resolver.is_owner = AsyncMock(return_value=False)
        msg = InboundMessage(
            channel="telegram",
            sender_id="u2",
            chat_id="c1",
            content="/restart",
            user_id="user-2",
        )
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

        with patch("krabobot.command.builtin.os.execv") as mock_execv:
            out = await cmd_restart(ctx)
            assert "только владелец" in out.content.lower()
            mock_execv.assert_not_called()

    @pytest.mark.asyncio
    async def test_status_intercepted_in_run_loop(self):
        """Verify /status is handled at the run-loop level for immediate replies."""
        loop, bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")

        with patch.object(loop, "_dispatch", new_callable=AsyncMock) as mock_dispatch:
            await bus.publish_inbound(msg)

            loop._running = True
            run_task = asyncio.create_task(loop.run())
            await asyncio.sleep(0.1)
            loop._running = False
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

            mock_dispatch.assert_not_called()
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert "krabobot" in out.content.lower() or "Model" in out.content

    @pytest.mark.asyncio
    async def test_run_propagates_external_cancellation(self):
        """External task cancellation should not be swallowed by the inbound wait loop."""
        loop, _bus = _make_loop()

        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0.1)
        run_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_help_includes_restart(self):
        loop, bus = _make_loop()
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/help")

        response = await loop._process_message(msg, runtime=loop._default_runtime)

        assert response is not None
        assert "/restart" in response.content
        assert "/status" in response.content
        assert "/id" in response.content
        assert "/clear_memory" in response.content
        assert response.metadata.get("render_as") == "text"
        assert response.metadata.get("_skip_tts") is True

    @pytest.mark.asyncio
    async def test_clear_memory_command_clears_current_runtime_memory(self):
        loop, _bus = _make_loop()
        runtime = loop._default_runtime
        runtime.memory_consolidator.store.clear_and_archive = MagicMock(return_value=True)
        msg = InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="c1",
            content="/clear_memory",
            user_id="u-test-owner",
        )

        response = await loop._process_message(msg, runtime=runtime)

        assert response is not None
        assert "Память очищена" in response.content
        runtime.memory_consolidator.store.clear_and_archive.assert_called_once()
        assert response.metadata.get("render_as") == "text"
        assert response.metadata.get("_skip_tts") is True

    @pytest.mark.asyncio
    async def test_id_command_returns_sender_and_chat_ids(self):
        loop, _bus = _make_loop()
        loop.user_resolver.accounts_for_user = AsyncMock(return_value=[])
        msg = InboundMessage(channel="telegram", sender_id="123|alice", chat_id="-1001", content="/id")

        response = await loop._process_message(msg, runtime=loop._default_runtime)

        assert response is not None
        assert "sender_id: 123|alice" in response.content
        assert "chat_id: -1001" in response.content
        assert "канал: telegram" in response.content
        assert response.metadata.get("render_as") == "text"
        assert response.metadata.get("_skip_tts") is True

    @pytest.mark.asyncio
    async def test_id_command_includes_all_linked_accounts(self):
        loop, _bus = _make_loop()
        loop.user_resolver.accounts_for_user = AsyncMock(
            return_value=["telegram:123|alice", "vk:85554821", "email:alice@example.com"]
        )
        msg = InboundMessage(
            channel="telegram",
            sender_id="123|alice",
            chat_id="-1001",
            content="/id",
            user_id="u-1",
        )

        response = await loop._process_message(msg)

        assert response is not None
        assert "Привязанные каналы:" in response.content
        assert "- telegram: 123|alice" in response.content
        assert "- vk: 85554821" in response.content
        assert "- email: alice@example.com" in response.content

    @pytest.mark.asyncio
    async def test_link_command_with_code_works_without_user_id(self):
        loop, _bus = _make_loop()
        loop.user_resolver.consume_link_code = AsyncMock(
            return_value=SimpleNamespace(ok=True, user_id="u-linked", error=None)
        )
        msg = InboundMessage(
            channel="telegram",
            sender_id="123|alice",
            chat_id="-1001",
            content="/link ABCD1234",
        )

        response = await loop._process_message(msg, runtime=loop._default_runtime)

        assert response is not None
        assert "Аккаунт успешно привязан" in response.content
        assert msg.user_id == "u-linked"

    @pytest.mark.asyncio
    async def test_tts_command_reports_status(self):
        loop, _bus = _make_loop()
        loop.user_resolver.get_tts_enabled = AsyncMock(return_value=False)
        msg = InboundMessage(
            channel="telegram",
            sender_id="123|alice",
            chat_id="-1001",
            content="/tts status",
            user_id="u-1",
        )

        response = await loop._process_message(msg)

        assert response is not None
        assert "TTS для вашего пользователя выключен" in response.content
        assert response.metadata.get("render_as") == "text"
        assert response.metadata.get("_skip_tts") is True

    @pytest.mark.asyncio
    async def test_status_reports_runtime_info(self):
        loop, _bus = _make_loop()
        loop.user_resolver.is_owner = AsyncMock(return_value=True)
        session = MagicMock()
        session.get_history.return_value = [{"role": "user"}] * 3
        loop.sessions.get_or_create.return_value = session
        loop._start_time = time.time() - 125
        loop._last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        loop.memory_consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(20500, "tiktoken")
        )

        msg = InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="c1",
            content="/status",
            user_id="u-test-owner",
        )

        response = await loop._process_message(msg, runtime=loop._default_runtime)

        assert response is not None
        assert "Модель: test-model" in response.content
        assert "Токены: 0 вход / 0 выход" in response.content
        assert "Контекст: 20k/64k (31%)" in response.content
        assert "Сессия: 3 сообщений" in response.content
        assert "Аптайм: 2m 5s" in response.content
        assert response.metadata.get("render_as") == "text"
        assert response.metadata.get("_skip_tts") is True

    @pytest.mark.asyncio
    async def test_status_denied_for_non_owner(self):
        loop, _bus = _make_loop()
        loop.user_resolver.is_owner = AsyncMock(return_value=False)
        msg = InboundMessage(
            channel="telegram",
            sender_id="u2",
            chat_id="c1",
            content="/status",
            user_id="user-2",
        )

        response = await loop._process_message(msg)

        assert response is not None
        assert "только владелец" in response.content.lower()
        assert response.metadata.get("render_as") == "text"
        assert response.metadata.get("_skip_tts") is True

    @pytest.mark.asyncio
    async def test_run_agent_loop_resets_usage_when_provider_omits_it(self):
        loop, _bus = _make_loop()
        loop.provider.chat_with_retry = AsyncMock(side_effect=[
            LLMResponse(content="first", usage={"prompt_tokens": 9, "completion_tokens": 4}),
            LLMResponse(content="second", usage={}),
        ])
        runtime = await loop._runtime_for_message(
            InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="hi")
        )

        await loop._run_agent_loop(runtime=runtime, initial_messages=[])
        assert loop._last_usage == {"prompt_tokens": 9, "completion_tokens": 4}

        await loop._run_agent_loop(runtime=runtime, initial_messages=[])
        assert loop._last_usage == {"prompt_tokens": 0, "completion_tokens": 0}

    @pytest.mark.asyncio
    async def test_status_falls_back_to_last_usage_when_context_estimate_missing(self):
        loop, _bus = _make_loop()
        loop.user_resolver.is_owner = AsyncMock(return_value=True)
        session = MagicMock()
        session.get_history.return_value = [{"role": "user"}]
        loop.sessions.get_or_create.return_value = session
        loop._last_usage = {"prompt_tokens": 1200, "completion_tokens": 34}
        loop.memory_consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(0, "none")
        )

        response = await loop._process_message(
            InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status", user_id="u-test-owner"),
            runtime=loop._default_runtime,
        )

        assert response is not None
        assert "Токены: 1200 вход / 34 выход" in response.content
        assert "Контекст: 1k/64k (1%)" in response.content

    @pytest.mark.asyncio
    async def test_process_direct_preserves_render_metadata(self):
        loop, _bus = _make_loop()
        session = MagicMock()
        session.get_history.return_value = []
        loop.sessions.get_or_create.return_value = session
        loop.subagents.get_running_count.return_value = 0

        response = await loop.process_direct("/status", session_key="cli:test")

        assert response is not None
        assert response.metadata.get("render_as") == "text"
        assert response.metadata.get("_skip_tts") is True

    @pytest.mark.asyncio
    async def test_unregistered_email_can_be_silenced_by_config(self):
        loop, _bus = _make_loop()
        loop.channels_config = SimpleNamespace(email={"replyRegisteredOnly": True})
        loop.user_resolver.is_registered = AsyncMock(return_value=False)
        msg = InboundMessage(
            channel="email",
            sender_id="new.user@example.com",
            chat_id="new.user@example.com",
            content="hello",
        )

        response = await loop._process_message(msg, runtime=loop._default_runtime)

        assert response is None
