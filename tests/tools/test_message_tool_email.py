"""Tests for message tool email recipient resolve/validation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from krabobot.agent.tools.message import (
    MessageTool,
    linked_email_addresses,
    resolve_email_chat_id,
)
from krabobot.utils.helpers import looks_like_email


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("andrey.tishkin@wefry.ru", True),
        ("a@b.co", True),
        ("e4fcddf8-6c08-4b20-b760-f35c3ed3d446", False),
        ("user_id", False),
        ("@nodomain", False),
        ("nodomain@", False),
        ("a@b", False),
        ("", False),
        (None, False),
    ],
)
def test_looks_like_email(value: str | None, expected: bool) -> None:
    assert looks_like_email(value) is expected


def test_linked_email_addresses_extracts_email_keys() -> None:
    assert linked_email_addresses(
        [
            "telegram:123",
            "email:alice@example.com",
            "email:bob@example.org",
            "email:not-an-email",
            "vk:99",
        ]
    ) == ["alice@example.com", "bob@example.org"]


@pytest.mark.asyncio
async def test_resolve_email_chat_id_keeps_valid_address() -> None:
    resolver = MagicMock()
    resolver.accounts_for_user = AsyncMock(return_value=[])
    got = await resolve_email_chat_id(
        "owner@wefry.ru",
        user_id="e4fcddf8-6c08-4b20-b760-f35c3ed3d446",
        user_resolver=resolver,
    )
    assert got == "owner@wefry.ru"
    resolver.accounts_for_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_email_chat_id_from_user_id_uuid() -> None:
    uid = "e4fcddf8-6c08-4b20-b760-f35c3ed3d446"
    resolver = MagicMock()
    resolver.accounts_for_user = AsyncMock(
        return_value=["email:andrey.tishkin@wefry.ru", "telegram:1"]
    )
    got = await resolve_email_chat_id(uid, user_id=uid, user_resolver=resolver)
    assert got == "andrey.tishkin@wefry.ru"


@pytest.mark.asyncio
async def test_resolve_email_chat_id_from_api_session_id() -> None:
    uid = "owner-user"
    session = "api-session-uuid-0001"
    resolver = MagicMock()
    resolver.accounts_for_user = AsyncMock(
        return_value=["email:owner@example.com", "vk:42"]
    )
    got = await resolve_email_chat_id(session, user_id=uid, user_resolver=resolver)
    assert got == "owner@example.com"


@pytest.mark.asyncio
async def test_resolve_email_chat_id_fails_without_linked_email() -> None:
    resolver = MagicMock()
    resolver.accounts_for_user = AsyncMock(return_value=["telegram:1"])
    got = await resolve_email_chat_id(
        "e4fcddf8-6c08-4b20-b760-f35c3ed3d446",
        user_id="u1",
        user_resolver=resolver,
    )
    assert got is None


@pytest.mark.asyncio
async def test_message_tool_resolves_uuid_to_linked_email() -> None:
    uid = "e4fcddf8-6c08-4b20-b760-f35c3ed3d446"
    resolver = MagicMock()
    resolver.accounts_for_user = AsyncMock(
        return_value=["email:andrey.tishkin@wefry.ru"]
    )
    resolver.lookup = AsyncMock(return_value=uid)
    resolver.get_tts_enabled = AsyncMock(return_value=False)
    outbound: list = []

    async def capture(m):
        outbound.append(m)

    tool = MessageTool(send_callback=capture, user_resolver=resolver)
    tool.set_context("api", "session-abc", user_id=uid)
    result = await tool.execute(
        content="test mail",
        channel="email",
        chat_id=uid,
    )

    assert not result.startswith("Error:")
    assert len(outbound) == 1
    assert outbound[0].channel == "email"
    assert outbound[0].chat_id == "andrey.tishkin@wefry.ru"
    assert "andrey.tishkin@wefry.ru" in result


@pytest.mark.asyncio
async def test_message_tool_uses_linked_email_when_chat_id_omitted_on_api() -> None:
    uid = "owner-1"
    resolver = MagicMock()
    resolver.accounts_for_user = AsyncMock(return_value=["email:owner@wefry.ru"])
    resolver.lookup = AsyncMock(return_value=uid)
    resolver.get_tts_enabled = AsyncMock(return_value=False)
    outbound: list = []

    async def capture(m):
        outbound.append(m)

    tool = MessageTool(send_callback=capture, user_resolver=resolver)
    tool.set_context("api", "session-xyz", user_id=uid)
    # Agent sets channel=email but omits chat_id → falls back to api session id
    result = await tool.execute(content="hello", channel="email")

    assert not result.startswith("Error:")
    assert outbound[0].chat_id == "owner@wefry.ru"


@pytest.mark.asyncio
async def test_message_tool_rejects_invalid_email_without_linked_account() -> None:
    resolver = MagicMock()
    resolver.accounts_for_user = AsyncMock(return_value=["telegram:1"])
    outbound: list = []

    async def capture(m):
        outbound.append(m)

    tool = MessageTool(send_callback=capture, user_resolver=resolver)
    tool.set_context("api", "session-xyz", user_id="u1")
    result = await tool.execute(
        content="hello",
        channel="email",
        chat_id="e4fcddf8-6c08-4b20-b760-f35c3ed3d446",
    )

    assert result.startswith("Error:")
    assert "valid recipient email" in result
    assert outbound == []
