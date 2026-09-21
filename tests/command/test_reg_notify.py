"""Tests for /reg owner notification when owner primary channel is api."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from krabobot.bus.events import InboundMessage, OutboundMessage
from krabobot.bus.queue import MessageBus
from krabobot.command.builtin import cmd_reg
from krabobot.command.router import CommandContext
from krabobot.session.manager import SessionManager
from krabobot.users import UserResolver


def _make_loop(tmp_path: Path):
    from krabobot.agent.loop import AgentLoop

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    with (
        patch("krabobot.agent.loop.ContextBuilder"),
        patch("krabobot.agent.loop.SubagentManager"),
    ):
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            session_manager=SessionManager(tmp_path),
        )
    return loop, bus


@pytest.mark.asyncio
async def test_reg_delivers_owner_notify_to_api_session(tmp_path: Path):
    """Owner linked on api gets registration notify in web session history."""
    loop, bus = _make_loop(tmp_path)
    resolver: UserResolver = loop.user_resolver

    owner = await resolver.resolve_or_create("api", "owner-web-session")
    await resolver.ensure_owner(owner)
    await resolver.link_account(owner, "cli", "user")

    msg = InboundMessage(
        channel="email",
        sender_id="sergey@example.com",
        chat_id="sergey@example.com",
        content="/reg Sergey",
    )
    ctx = CommandContext(
        msg=msg,
        session=None,
        key=msg.session_key,
        raw="/reg Sergey",
        args="Sergey",
        loop=loop,
    )

    out = await cmd_reg(ctx)

    assert "Заявка на регистрацию отправлена" in out.content
    assert bus.outbound_size == 0  # api/cli must not go through the bus

    sm = await loop.session_manager_for_api("owner-web-session")
    session = sm.get_or_create("api:owner-web-session")
    assistant = [m for m in session.messages if m.get("role") == "assistant"]
    assert assistant, "expected owner notify in api session"
    assert "Новая заявка на регистрацию" in assistant[-1]["content"]
    assert "/reg approve" in assistant[-1]["content"]


@pytest.mark.asyncio
async def test_reg_also_publishes_to_email_when_owner_has_email(tmp_path: Path):
    """Owner with api + email gets session write and bus publish for email."""
    loop, bus = _make_loop(tmp_path)
    resolver: UserResolver = loop.user_resolver

    owner = await resolver.resolve_or_create("api", "owner-sess")
    await resolver.ensure_owner(owner)
    await resolver.link_account(owner, "email", "owner@example.com")

    msg = InboundMessage(
        channel="email",
        sender_id="newbie@example.com",
        chat_id="newbie@example.com",
        content="/reg",
    )
    ctx = CommandContext(
        msg=msg, session=None, key=msg.session_key, raw="/reg", args="", loop=loop
    )

    out = await cmd_reg(ctx)
    assert "Заявка на регистрацию отправлена" in out.content

    # email outbound on bus
    assert bus.outbound_size >= 1
    email_msg = await bus.consume_outbound()
    assert email_msg.channel == "email"
    assert "Новая заявка" in email_msg.content

    sm = await loop.session_manager_for_api("owner-sess")
    session = sm.get_or_create("api:owner-sess")
    assert any(
        m.get("role") == "assistant" and "Новая заявка" in str(m.get("content", ""))
        for m in session.messages
    )


@pytest.mark.asyncio
async def test_reg_still_returns_when_deliver_outbound_raises(tmp_path: Path):
    """Requester always gets a confirmation even if owner notify blows up."""
    loop, bus = _make_loop(tmp_path)
    resolver: UserResolver = loop.user_resolver
    owner = await resolver.resolve_or_create("api", "own")
    await resolver.ensure_owner(owner)

    loop.deliver_outbound = AsyncMock(side_effect=RuntimeError("boom"))

    msg = InboundMessage(
        channel="email",
        sender_id="x@y.z",
        chat_id="x@y.z",
        content="/reg",
    )
    ctx = CommandContext(
        msg=msg, session=None, key=msg.session_key, raw="/reg", args="", loop=loop
    )
    out = await cmd_reg(ctx)
    assert "Заявка на регистрацию отправлена" in out.content


@pytest.mark.asyncio
async def test_deliver_outbound_api_does_not_warn_unknown_channel(tmp_path: Path):
    loop, _bus = _make_loop(tmp_path)
    owner = await loop.user_resolver.resolve_or_create("api", "s1")
    await loop.user_resolver.ensure_owner(owner)

    ok = await loop.deliver_outbound(
        OutboundMessage(channel="api", chat_id="s1", content="hello owner")
    )
    assert ok is True
    sm = await loop.session_manager_for_api("s1")
    session = sm.get_or_create("api:s1")
    assert session.messages[-1]["content"] == "hello owner"
