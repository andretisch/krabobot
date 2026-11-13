"""LDAP web chat must be registered without a second /reg.

Password/adminToken login leaves user_id empty; serve auto-links api:{session}
to the owner. LDAP login scopes the turn with user_id + _preserve_owner, and
used to skip that link — is_registered(api, session_id) stayed false.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from krabobot.agent.loop import AgentLoop
from krabobot.bus.queue import MessageBus
from krabobot.providers.base import GenerationSettings, LLMResponse


def _make_loop(tmp_path: Path) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=512)
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="привет", tool_calls=[])
    )
    provider.chat_stream_with_retry = AsyncMock(
        return_value=LLMResponse(content="привет", tool_calls=[])
    )
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )


@pytest.mark.asyncio
async def test_ldap_scoped_api_chat_is_registered_without_reg(tmp_path: Path) -> None:
    """Existing user (owner or guest) logging in via LDAP must chat on api:*."""
    loop = _make_loop(tmp_path)
    owner = await loop.user_resolver.create_user(
        display_name="Andrey",
        channel="telegram",
        sender_id="1001",
    )
    await loop.user_resolver.ensure_owner(owner)
    await loop.user_resolver.link_ldap(owner, "andrey.tishkin")

    session_id = "776ba3f4-c477-456b-b992-c5072725ecb0"
    assert await loop.user_resolver.is_registered("api", session_id) is False

    out = await loop.process_direct(
        "Как см?",
        session_key=f"api:{session_id}",
        channel="api",
        chat_id=session_id,
        sender_id=session_id,
        user_id=owner,
    )

    assert out is not None
    assert "Доступ не активирован" not in (out.content or "")
    assert out.content == "привет"
    assert await loop.user_resolver.is_registered("api", session_id) is True
    assert await loop.user_resolver.lookup("api", session_id) == owner
    assert await loop.user_resolver.lookup_ldap("andrey.tishkin") == owner


@pytest.mark.asyncio
async def test_ldap_scoped_api_chat_links_guest_without_claiming_owner(
    tmp_path: Path,
) -> None:
    loop = _make_loop(tmp_path)
    owner = await loop.user_resolver.create_user(
        display_name="Owner",
        channel="telegram",
        sender_id="1",
    )
    await loop.user_resolver.ensure_owner(owner)
    guest = await loop.user_resolver.create_user(
        display_name="Sergey",
        channel="telegram",
        sender_id="354021402",
    )
    await loop.user_resolver.link_ldap(guest, "sergey.ivannikov")

    session_id = "web-ldap-guest-sess"
    out = await loop.process_direct(
        "как дела",
        session_key=f"api:{session_id}",
        channel="api",
        chat_id=session_id,
        sender_id=session_id,
        user_id=guest,
    )

    assert out is not None
    assert "Доступ не активирован" not in (out.content or "")
    assert await loop.user_resolver.lookup("api", session_id) == guest
    assert await loop.user_resolver.get_owner_user_id() == owner


@pytest.mark.asyncio
async def test_scoped_user_id_does_not_autolink_voice(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    owner = await loop.user_resolver.create_user(
        display_name="Owner",
        channel="telegram",
        sender_id="1",
    )
    await loop.user_resolver.ensure_owner(owner)

    out = await loop.process_direct(
        "ping",
        session_key="voice:pi-01",
        channel="voice",
        chat_id="pi-01",
        sender_id="pi-01",
        user_id=owner,
    )

    assert out is not None
    assert "Доступ не активирован" in (out.content or "")
    assert await loop.user_resolver.is_registered("voice", "pi-01") is False
